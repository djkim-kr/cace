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
                 bec_key: str = 'LES_BEC',
                 compute_energy: bool = True,
                 compute_bec: bool = False,
                 bec_output_index: int = None, # option to compute BEC along one axis
                 ):
        super().__init__()
        from les import Les
        use_les_atomwise = True
        if feature_key is None:
            # directly provide the charges to LES
            use_les_atomwise = False
        self.les = Les(les_arguments={"use_atomwise": use_les_atomwise})
 
        self.feature_key = feature_key
        self.energy_key = energy_key
        self.charge_key = charge_key
        self.dipole_key = dipole_key
        self.kappa_key = kappa_key
        self.alpha_key = alpha_key

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

    def set_compute_bec(self, compute_bec: bool):
        self.compute_bec = compute_bec

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

        result = self.les(
            desc=features,
            latent_charges=data[self.charge_key] if features is None else None,
            latent_dipoles=data[self.dipole_key] if self.dipole_key is not None else None,
            latent_alphas=data[self.alpha_key] if self.alpha_key is not None else None,
            latent_kappas=data[self.kappa_key] if self.kappa_key is not None else None,
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

        if self.compute_energy:
            data[self.energy_key] = result['E_lr']
        if self.compute_bec:
            data[self.bec_key] = result['BEC']
        return data
