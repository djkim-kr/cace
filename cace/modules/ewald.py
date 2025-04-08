import torch
import torch.nn as nn
from itertools import product
from typing import Dict, Optional, Tuple
import numpy as np

class EwaldPotential(nn.Module):
    def __init__(self,
                 dl=2.0,  # grid resolution
                 sigma=1.0,  # width of the Gaussian on each atom
                 exponent=1, # default is for electrostattics with p=1, we can do London dispersion with p=6
                 external_field = None, # external field
                 external_field_direction: int = 0, # external field direction, 0 for x, 1 for y, 2 for z
                 charge_neutral_lambda: float = None,
                 remove_self_interaction=False,
                 feature_key: str = 'q',
                 output_key: str = 'ewald_potential',
                 aggregation_mode: str = "sum",
                 compute_field: bool = False,
                 ):
        super().__init__()
        self.dl = dl
        self.sigma = sigma
        self.exponent = exponent
        self.sigma_sq_half = sigma ** 2 / 2.0
        self.twopi = 2.0 * torch.pi
        self.twopi_sq = self.twopi ** 2
        self.remove_self_interaction = remove_self_interaction
        self.feature_key = feature_key
        self.output_key = output_key
        self.aggregation_mode = aggregation_mode
        self.model_outputs = [output_key]
        # 1/2\epsilon_0, where \epsilon_0 is the vacuum permittivity
        # \epsilon_0 = 5.55263*10^{-3} e^2 eV^{-1} A^{-1}
        #self.norm_factor = 90.0474
        self.norm_factor = 1.0 
        # when using a norm_factor = 1, all "charges" are scaled by sqrt(90.0474)
        # the external field is then scaled by sqrt(90.0474) = 9.48933
        self.k_sq_max = (self.twopi / self.dl) ** 2
        self.external_field = external_field
        self.external_field_direction = external_field_direction
        self.compute_field = compute_field
        if self.compute_field:
            self.model_outputs.append(feature_key+'_field')

        self.charge_neutral_lambda = charge_neutral_lambda

    def forward(self, data: Dict[str, torch.Tensor],
                training: bool = None,
                output_index: Optional[int]=None,
                ) -> Dict[str, torch.Tensor]:
        
        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch_now = data["batch"]

        # box = data['cell'].view(-1, 3, 3).diagonal(dim1=-2, dim2=-1)
        box = data['cell'].view(-1, 3, 3)
        r = data['positions']
        q = data[self.feature_key]
        if q.dim() == 1:
            q = q.unsqueeze(1)

        # Check the input dimension
        n, d = r.shape
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'

        unique_batches = torch.unique(batch_now)  # Get unique batch indices

        # this is just for compatibility with the previous version
        if hasattr(self, 'exponent') == False:
            self.exponent = 1
        if hasattr(self, 'compute_field') == False:
            self.compute_field = False
        #torch.jit.fork/wait -> asyncchrounously GPU kernel(parallel)
        futures = [torch.jit.fork(self._compute_for_configuration,
                                    r[batch_now == b],
                                    q[batch_now == b],
                                    box[b],
                                    data,
                                    int(b))
                for b in unique_batches]
        batch_results = [torch.jit.wait(f) for f in futures]

        pot_list = [result[0] for result in batch_results]
        field_list = [result[1] for result in batch_results]

        data[self.output_key] = torch.stack(pot_list, dim=0).sum(dim=1)
        if self.compute_field:
            data[self.feature_key+'_field'] = torch.cat(field_list, dim=0)
        return data

    def _compute_for_configuration(self,
                               r_config: torch.Tensor,
                               q_config: torch.Tensor,
                               box_config: torch.Tensor,
                               data: Dict[str, torch.Tensor],
                               batch_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        box_diag = box_config.diagonal(dim1=-2, dim2=-1)
        if box_diag[0] < 1e-6 and box_diag[1] < 1e-6 and box_diag[2] < 1e-6 and self.exponent == 1:
            # the box is not periodic, we use the direct sum
            pot, field = self.compute_potential_realspace(r_config, q_config, self.compute_field)
        elif box_diag[0] > 0 and box_diag[1] > 0 and box_diag[2] > 0:
            # the box is periodic, we use the reciprocal sum
            pot, field = self.compute_potential_triclinic(r_config, q_config, box_config, self.compute_field)
        else:
            raise ValueError("Either all box dimensions must be positive or aperiodic box must be provided.")
        
        if self.exponent == 1 and hasattr(self, 'external_field') and self.external_field is not None:
            # if self.external_field_direction is an integer, then external_field_direction is the direction index
            if isinstance(self.external_field_direction, int):
                direction_index_now = self.external_field_direction
                # if self.external_field_direction is a string, then it is the key to the external field
            elif self.external_field_direction in data and data[self.external_field_direction] is not None:
                direction_index_now = int(data[self.external_field_direction][batch_index])
            else:
                raise ValueError("external_field_direction must be an integer or a key to the external field")
            if isinstance(self.external_field, float):
                external_field_now = self.external_field
            elif self.external_field in data and data[self.external_field] is not None:
                external_field_now = data[self.external_field][batch_index]
            else:
                raise ValueError("external_field must be a float or a key to the external field")
            box_now = box_config.diagonal(dim1=-2, dim2=-1)
            pot_ext = self.add_external_field(r_config, q_config, box_now, direction_index_now, external_field_now)
        else:
            pot_ext = 0.0

        if hasattr(self, 'charge_neutral_lambda') and self.charge_neutral_lambda is not None:
            q_mean = torch.mean(q_config)
            pot_neutral = self.charge_neutral_lambda * (q_mean)**2.
        else:
            pot_neutral = 0.0

        return pot + pot_ext + pot_neutral, field

    def compute_potential_realspace(self, r_raw, q, compute_field:bool =False):
        # Compute pairwise distances (norm of vector differences)
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)
        r_ij_norm = torch.norm(r_ij, dim=-1)
        #print(r_ij_norm)
 
        # Error function scaling for long-range interactions
        convergence_func_ij = torch.special.erf(r_ij_norm / self.sigma / (2.0 ** 0.5))
        #print(convergence_func_ij)
   
        # Compute inverse distance safely
        # [n_node, n_node]
        #r_p_ij = torch.where(r_ij_norm > 1e-3, 1.0 / r_ij_norm, 0.0) # this causes gradient issues
        epsilon = 1e-6
        r_p_ij = 1.0 / (r_ij_norm + epsilon)

        if q.dim() == 1:
            # [n_node, n_q]
            q = q.unsqueeze(1)
    
        # Compute potential energy
        n_node, n_q = q.shape
        # Use broadcasting to set diagonal elements to 0
        #mask = torch.ones(n_node, n_node, n_q, dtype=torch.int64, device=q.device)
        #diag_indices = torch.arange(n_node)
        #mask[diag_indices, diag_indices, :] = 0
        # [1, n_node, n_q] * [n_node, 1, n_q] * [n_node, n_node, 1] * [n_node, n_node, 1]
        pot = torch.sum(q.unsqueeze(0) * q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2)).view(-1) / self.twopi / 2.0
    
        q_field = torch.zeros_like(q, dtype=q.dtype, device=q.device) # Field due to q
        # Compute field if requested
        if compute_field:
            # [n_node, 1 , n_q] * [n_node, n_node, 1] * [n_node, n_node, 1]
            q_field = torch.sum(q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2), dim=0) / self.twopi

        # because this realspace sum already removed self-interaction, we need to add it back if needed
        if self.remove_self_interaction == False and self.exponent == 1:
            pot += torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))
            q_field = q_field + q / (self.sigma * self.twopi**(3./2.)) * 2.
    
        return pot * self.norm_factor, q_field * self.norm_factor
 
    def add_external_field(self, r_raw, q, box, direction_index, external_field):
        external_field_norm_factor = (self.norm_factor/90.0474)**0.5
        # wrap in box
        r = r_raw[:, direction_index] / box[direction_index]
        r =  r - torch.round(r)
        r = r * box[direction_index]
        return external_field * torch.sum(q * r.unsqueeze(1)) * external_field_norm_factor

    def change_external_field(self, external_field):
        self.external_field = external_field

    def is_orthorhombic(self, cell_matrix):
        diag_matrix = torch.diag(torch.diagonal(cell_matrix))
        is_orthorhombic = torch.allclose(cell_matrix, diag_matrix, atol=1e-6)
        return is_orthorhombic
    
    # Triclinic box(could be orthorhombic)
    def compute_potential_triclinic(self, r_raw, q, cell_now, compute_field: bool=False):
        device = r_raw.device

        cell_inv = torch.linalg.inv(cell_now)
        G = 2 * torch.pi * cell_inv.T  # Reciprocal lattice vectors [3,3], G = 2π(M^{-1}).T

        # max Nk for each axis
        norms = torch.norm(cell_now, dim=1)
        Nk = [max(1, int(n.item() / self.dl)) for n in norms]
        n1 = torch.arange(-Nk[0], Nk[0] + 1, device=device)
        n2 = torch.arange(-Nk[1], Nk[1] + 1, device=device)
        n3 = torch.arange(-Nk[2], Nk[2] + 1, device=device)

        # Create nvec grid and compute k vectors
        nvec = torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3)
        nvec = nvec.to(G.dtype)
        # kvec = G @ nvec
        kvec = (nvec.float() @ G).to(device)  # [N_total, 3]

        # Apply k-space cutoff and filter
        k_sq = torch.sum(kvec ** 2, dim=1)
        mask = (k_sq > 0) & (k_sq <= self.k_sq_max)
        kvec = kvec[mask] # [M, 3]
        k_sq = k_sq[mask] # [M]
        nvec = nvec[mask] # [M, 3]

        # Determine symmetry factors (handle hemisphere to avoid double-counting)
        # Include nvec if first non-zero component is positive
        non_zero = (nvec != 0).to(torch.int)
        first_non_zero = torch.argmax(non_zero, dim=1)
        sign = torch.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze()
        hemisphere_mask = (sign > 0) | ((nvec == 0).all(dim=1))
        kvec = kvec[hemisphere_mask]
        k_sq = k_sq[hemisphere_mask]
        factors = torch.where((nvec[hemisphere_mask] == 0).all(dim=1), 1.0, 2.0)

        # Compute structure factor S(k), Σq*e^(ikr)
        k_dot_r = torch.matmul(r_raw, kvec.T)  # [n, M]
        exp_ikr = torch.exp(1j * k_dot_r)
        S_k = torch.sum(q * exp_ikr, dim=0)  # [M]

        #for torchscript compatibility, to avoid dtype mismatch, only use real part
        cos_k_dot_r = torch.cos(k_dot_r)
        sin_k_dot_r = torch.sin(k_dot_r)
        S_k_real = torch.sum(q * cos_k_dot_r, dim=0)  # [M]
        S_k_imag = torch.sum(q * sin_k_dot_r, dim=0)  # [M]
        S_k_sq = S_k_real**2 + S_k_imag**2  # [M]

        # Compute kfac,  exp(-σ^2/2 k^2) / k^2 for exponent = 1
        if self.exponent == 1:
            kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
        elif self.exponent == 6:
            b_sq = k_sq * self.sigma_sq_half
            b = torch.sqrt(b_sq)
            kfac = -1.0 * k_sq**(3/2) * (
                torch.sqrt(torch.tensor(torch.pi)) * torch.special.erfc(b) + 
                (1/(2*b**3) - 1/b) * torch.exp(-b_sq)
            )
        else:
            raise ValueError("Exponent must be 1 or 6")
        
        # Compute potential, (2π/volume)* sum(factors * kfac * |S(k)|^2)
        volume = torch.det(cell_now)
        pot = (factors * kfac * S_k_sq).sum() / volume
        
        # Compute electric field if needed
        q_field = torch.zeros_like(q, dtype=r_raw.dtype, device=device)
        if compute_field:
            sk_field = 2 * kfac * torch.conj(S_k)
            q_field = (factors * torch.real(exp_ikr * sk_field)).sum(dim=1) / volume

        # Remove self-interaction if applicable
        if self.remove_self_interaction and self.exponent == 1:
            pot -= torch.sum(q**2) / (self.sigma * (2 * torch.pi)**1.5)
            q_field -= q * (2 / (self.sigma * (2 * torch.pi)**1.5))

        return pot.unsqueeze(0) * self.norm_factor, q_field.unsqueeze(1) * self.norm_factor
