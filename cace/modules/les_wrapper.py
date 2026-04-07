import torch
import torch.nn as nn
from typing import Dict, Sequence, Union

__all__ = ['LesWrapper']

class LesWrapper(nn.Module):
    """
    A wrapper for the LES library that does long-range interactions and BECs
    Note that CACE has its own internal implementation of the LES algorithm
    so it is not necessary to use this wrapper in CACE.
    """
    def __init__(self,
                 feature_key: Union[str, Sequence[int]] = 'node_feats',
                 energy_key: str = 'LES_energy',
                 charge_key: str = 'LES_charge',
                 dipole_key: str = None,
                 kappa_key: str = None,
                 alpha_key: str = None,
                 atomic_number_key: str = None,
                 bec_key: str = 'LES_BEC',
                 compute_energy: bool = True,
                 compute_bec: bool = False,
                 bec_output_index: int = None, # option to compute BEC along one axis
                 make_alpha_positive: bool = False,
                 make_kappa_positive: bool = False,
                 add_scalar_alpha: bool = False,
                 use_atomic_alpha: bool = False,
                 use_epsilon_r_scaling: bool = False,
                 ):
        super().__init__()
        from les import Les
        use_les_atomwise = True
        if feature_key is None:
            # directly provide the charges to LES
            use_les_atomwise = False
        self.les = Les(les_arguments={"use_atomwise": use_les_atomwise, 
                                      "use_atomic_alpha": use_atomic_alpha,
                                      "use_epsilon_r_scaling": use_epsilon_r_scaling,
                                      })
 
        self.feature_key = feature_key
        self.energy_key = energy_key
        self.charge_key = charge_key
        self.dipole_key = dipole_key
        self.kappa_key = kappa_key
        self.alpha_key = alpha_key
        self.atomic_number_key = atomic_number_key

        self.make_alpha_positive = make_alpha_positive
        self.make_kappa_positive = make_kappa_positive
        self.add_scalar_alpha = add_scalar_alpha
        if self.add_scalar_alpha:
            from les.module import Atomwise
            self.atomwise: nn.Module = (
                Atomwise(
                    n_layers=3,
                    n_hidden=[32,16],
                    add_linear_nn=True,
                    output_scaling_factor=1, 
                )
            )

        self.bec_key = bec_key
        self.bec_output_index = bec_output_index

        self.compute_energy = compute_energy        
        self.compute_bec = compute_bec
        self.model_outputs = [charge_key]
        if compute_energy:
            self.model_outputs.append(energy_key)
        if compute_bec:
            self.model_outputs.append(bec_key)
        self.required_derivatives = []
        self.required_derivatives.append('cell')

    def set_compute_energy(self, compute_energy: bool):
        self.compute_energy = compute_energy
        if compute_energy and self.energy_key not in self.model_outputs:
            self.model_outputs.append(self.energy_key)

    def set_compute_bec(self, compute_bec: bool):
        self.compute_bec = compute_bec
        if compute_bec and self.bec_key not in self.model_outputs:
            self.model_outputs.append(self.bec_key)

    def set_bec_output_index(self, bec_output_index: int):
        self.bec_output_index = bec_output_index

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:

        # if charge key is already in data, we use the provided charges and skip the LES charge prediction
        if self.charge_key in data:
            features = None
        # reshape the feature vectors
        elif isinstance(self.feature_key, str):
            if self.feature_key not in data:
                raise ValueError(f"Feature key {self.feature_key} not found in data dictionary.")
            features = data[self.feature_key]
            features = features.reshape(features.shape[0], -1)
        elif isinstance(self.feature_key, list):
            features = torch.cat([data[key].reshape(data[key].shape[0], -1) for key in self.feature_key], dim=-1)

        if hasattr(self, 'add_scalar_alpha') and self.add_scalar_alpha:
            alpha = data[self.alpha_key] if self.alpha_key in data else None
            if alpha.dim() == 3 and alpha.shape[1] == 3 and alpha.shape[2] == 3:
                desc = data[self.feature_key]
                a2 = self.atomwise(desc.reshape(desc.shape[0],-1),data["batch"]).squeeze()
                eye = torch.eye(3,device=a2.device)
                a2 = a2[:,None,None] * eye[None,:,:]
                data[self.alpha_key] = data[self.alpha_key] + a2

        if hasattr(self, 'make_alpha_positive') and self.make_alpha_positive:
            alpha = data[self.alpha_key] if self.alpha_key in data else None
            if alpha.dim() == 2:
                data[self.alpha_key] = alpha**2
            if alpha.dim() == 3 and alpha.shape[1] == 3 and alpha.shape[2] == 3:
                data[self.alpha_key] = torch.einsum("nij,nkj->nik",alpha, alpha)
        if hasattr(self, 'make_kappa_positive') and self.make_kappa_positive:
            data[self.kappa_key] = data[self.kappa_key]**2

        result = self.les(
            desc=features,
            latent_charges=data[self.charge_key] if features is None else None,
            latent_dipoles=data[self.dipole_key] if self.dipole_key is not None else None,
            latent_alphas=data[self.alpha_key] if self.alpha_key is not None else None,
            latent_kappas=data[self.kappa_key] if self.kappa_key is not None else None,
            atomic_numbers=data[self.atomic_number_key] if hasattr(self, 'atomic_number_key') and  self.atomic_number_key is not None else None,
            positions=data['positions'],
            cell=data['cell'].view(-1, 3, 3),
            batch=data["batch"],
            compute_energy=self.compute_energy,
            compute_bec=self.compute_bec,
            bec_output_index=self.bec_output_index,
        )

        # update the data dictionary with the results
        data[self.charge_key] = result['latent_charges']
        if self.dipole_key is not None:
            data[self.dipole_key] = result['latent_dipoles']
        if self.alpha_key is not None:
            data[self.alpha_key] = result['latent_alphas']

        if self.compute_energy:
            data[self.energy_key] = result['E_lr']
        if self.compute_bec:
            data[self.bec_key] = result['BEC']
        return data
