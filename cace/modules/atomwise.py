from typing import Dict, Union, Sequence, Callable, Optional
from unicodedata import name

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import Dense, ResidualBlock, build_mlp
from ..tools import scatter_sum

__all__ = ["Atomwise", "Atomwise_linear"]

class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.

    If `aggregation_mode` is None, only the per-atom predictions will be returned.
    """

    def __init__(
        self,
        n_in: Optional[int] = None,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        bias: bool = True,
        activation: Callable = F.silu,
        aggregation_mode: str = "sum",
        feature_key: Union[str, Sequence[int]] = 'node_feats',
        output_key: str = "energy",
        per_atom_output_key: Optional[str] = None,
        descriptor_output_key: Optional[str] = None,
        residual: bool = False,
        use_batchnorm: bool = False,
        add_linear_nn: bool = False,
        post_process: Optional[Callable] = None,
        per_atom_output_key_2: Optional[str] = None,  # for multi-head output
        energy_output_index: Optional[int] = None,  # for multi-head output
        charge_output_index: Optional[int] = None,  # for multi-head output
    ):
        """
        Args:
            n_in: input dimension of representation
            n_out: output dimension of target property (default: 1)
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers resulting
                in a rectangular network.
                If None, the number of neurons is divided by two after each layer starting
                n_in resulting in a pyramidal network.
            n_layers: number of layers.
            aggregation_mode: one of {sum, avg} (default: sum)
            output_key: the key under which the result will be stored
            per_atom_output_key: If not None, the key under which the per-atom result will be stored
            residual: whether to use residual connections between layers
            use_batchnorm: whether to use batch normalization between layers
            add_linear_nn: whether to add a linear NN to the output of the MLP 
        """
        super().__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.per_atom_output_key = per_atom_output_key
        self.descriptor_output_key = descriptor_output_key
        if self.per_atom_output_key is not None:
            self.model_outputs.append(self.per_atom_output_key)
        self.per_atom_output_key_2 = per_atom_output_key_2
        if per_atom_output_key_2 is not None:
            self.model_outputs.append(per_atom_output_key_2)
        if self.descriptor_output_key is not None: 
            self.model_outputs.append(self.descriptor_output_key)

        self.n_out = n_out

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.aggregation_mode = aggregation_mode
        self.residual = residual
        self.use_batchnorm = use_batchnorm
        self.add_linear_nn = add_linear_nn
        self.post_process = post_process
        self.bias = bias
        self.feature_key = feature_key
        if self.n_out == 2:
            self.energy_output_index = energy_output_index
            self.charge_output_index = charge_output_index
        else:
            self.energy_output_index = None
            self.charge_output_index = None

        if n_in is not None:
            self.outnet = build_mlp(
                n_in=self.n_in,
                n_out=self.n_out,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers,
                activation=self.activation,
                residual=self.residual,
                use_batchnorm=self.use_batchnorm,
                bias=self.bias,
                )
            if self.add_linear_nn:
                self.linear_nn = Dense(
                   self.n_in, 
                   self.n_out,
                   bias=self.bias,
                   activation=None, 
                   use_batchnorm=self.use_batchnorm,
                   ) 

        else:
            self.outnet = None

    def forward(self, 
                data: Dict[str, torch.Tensor], 
                training: bool = None,
                output_index: int = None, # only used for multi-head output
               ) -> Dict[str, torch.Tensor]:

        # check if self.feature_key exists, otherwise set default 
        if not hasattr(self, "feature_key") or self.feature_key is None: 
            self.feature_key = "node_feats"

        # reshape the feature vectors
        if isinstance(self.feature_key, str):
            if self.feature_key not in data:
                raise ValueError(f"Feature key {self.feature_key} not found in data dictionary.")
            features = data[self.feature_key]
            features = features.reshape(features.shape[0], -1)
        elif isinstance(self.feature_key, list):
            features = torch.cat([data[key].reshape(data[key].shape[0], -1) for key in self.feature_key], dim=-1)

        if self.n_in is None:
            self.n_in = features.shape[1]
        else:
            assert self.n_in == features.shape[1]

        if self.outnet == None:
            self.outnet = build_mlp(
                n_in=self.n_in,
                n_out=self.n_out,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers,
                activation=self.activation,
                residual=self.residual,
                use_batchnorm=self.use_batchnorm,
                bias=self.bias,
                )
            self.outnet = self.outnet.to(features.device)
            if self.add_linear_nn:
                self.linear_nn = Dense(
                   self.n_in,
                   self.n_out,
                   bias=self.bias,
                   activation=None,
                   use_batchnorm=self.use_batchnorm,
                   )
                self.linear_nn = self.linear_nn.to(features.device)
            else:
                self.linear_nn = None

        # predict atomwise contributions
        y = self.outnet(features)
        if self.add_linear_nn:
            y += self.linear_nn(features)

        # accumulate the per-atom output if necessary
        if self.per_atom_output_key is not None:
            if self.energy_output_index is not None:
                a = y[:, self.energy_output_index]
                data[self.per_atom_output_key] = a

        if self.per_atom_output_key_2 is not None:
            if self.charge_output_index is not None:
                a = y[:, self.charge_output_index]
                data[self.per_atom_output_key_2] = a

        if hasattr(self, "descriptor_output_key") and self.descriptor_output_key is not None:
            data[self.descriptor_output_key] = features

        # aggregate
        if self.aggregation_mode is not None:
            y = scatter_sum(
                src=y, 
                index=data["batch"], 
                dim=0)
            y = torch.squeeze(y, -1)

            if self.aggregation_mode == "avg":
                y = y / torch.bincount(data['batch'])

        if hasattr(self, "post_process") and self.post_process is not None:
            y = self.post_process(y)
        data[self.output_key] = y[:, self.energy_output_index] if self.energy_output_index is not None else y

        # for name, params in self.named_parameters():
        #     print (f"Parameter: {name}, requires_grad: {params.requires_grad}, shape: {params.shape}")
        #     if params.dim() >= 2:
        #         print(params[0,0])
        #     else:
        #         print (params[0])

        return data

class Atomwise_linear(nn.Module):
    """
    testing purpose
    """

    def __init__(
            self,
            n_in: Optional[int] = None,
            n_out: int = 1,
            bias: bool = True,
            activation: Callable = F.silu,
            feature_key: Union[str, Sequence[int]] = 'node_feats',
            per_atom_output_key: Optional[str] = None,
            use_batchnorm: bool = False,
            post_process: Optional[Callable] = None,
            output_index: int = None, # only used for multi-head output
        ):
            super().__init__()
            self.model_outputs = []
            self.per_atom_output_key = per_atom_output_key
            if self.per_atom_output_key is not None:
                self.model_outputs.append(self.per_atom_output_key)

            self.n_out = n_out

            self.n_in = n_in
            self.n_out = n_out
            self.activation = activation
            self.use_batchnorm = use_batchnorm
            self.post_process = post_process
            self.bias = bias
            self.feature_key = feature_key
            self.outnet = None
            self.output_index = output_index

    def forward(self, 
                data: Dict[str, torch.Tensor], 
                training: bool = None,
                output_index: int = None, # only used for multi-head output
               ) -> Dict[str, torch.Tensor]:
        
        if not hasattr(self, "feature_key") or self.feature_key is None: 
            self.feature_key = "node_feats"

        if isinstance(self.feature_key, str):
            if self.feature_key not in data:
                raise ValueError(f"Feature key {self.feature_key} not found in data dictionary.")
            features = data[self.feature_key]
            features = features.reshape(features.shape[0], -1)
        elif isinstance(self.feature_key, list):
            features = torch.cat([data[key].reshape(data[key].shape[0], -1) for key in self.feature_key], dim=-1)

        if self.n_in is None:
            self.n_in = features.shape[1]
        else:
            assert self.n_in == features.shape[1]

        self.outnet = nn.Linear(self.n_in, self.n_out, self.bias)

        # predict atomwise contributions
        y = self.outnet(features)

        # accumulate the per-atom output if necessary
        if self.per_atom_output_key is not None:
            if hasattr(self, "post_process") and self.post_process is not None:
                y = self.post_process(y)
            data[self.per_atom_output_key] = y
        
        return data