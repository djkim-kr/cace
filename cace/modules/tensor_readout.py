from typing import Dict
import torch
from torch import nn

from .tensornet import TensorFeedForward

__all__ = ['TensorReadout']

class TensorReadout(nn.Module):
    """
    Predicts dipoles/quadrupoles directly from node_feats_l
    """

    def __init__(
        self,
        max_l: int,
        feature_key: str = 'node_feats_l',
        l0_key: str = 'scalar',
        l1_key: str = 'vector',
        l2_key: str = 'quadrupole',
        n_channel: int = 1,
    ):
        """
        Args:        
        """
        super().__init__()
        assert max_l >= 1, "max_l must be at least 1 for vector prediction."
        assert max_l <= 2, "max_l greater than 2 is not supported for direct prediction."
        self.max_l = max_l
        self.feature_key = feature_key
        self.l0_key = l0_key
        self.l1_key = l1_key
        self.l2_key = l2_key

        self.model_outputs = []
        self.model_outputs.append(self.l1_key)
        if max_l >= 2:
            self.model_outputs.append(self.l2_key)

        self.required_derivatives = []

        self.tensor_feed_forward = TensorFeedForward(n_channel, lomax=max_l)

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        if self.feature_key not in data:
            raise ValueError(f"Feature key {self.feature_key} not found in data dictionary.")
        features = data[self.feature_key] #{0: l=0, 1:l=1, 2:l=2...}

        assert 1 in features, f"Features for l=1 not found in data dictionary under key {self.feature_key}."
        if self.max_l >= 2:
            assert 2 in features, f"Features for l=2 not found in data dictionary under key {self.feature_key}."

        out = self.tensor_feed_forward(features)
        data[self.l1_key] = out[1][:,0]
        if self.max_l >= 2:
            data[self.l2_key] = out[2][:,0]

        return data 

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            )
