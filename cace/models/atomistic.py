from typing import Dict, Optional, List

import torch
import torch.nn as nn

from ..modules import Transform
from ..tools import torch_geometric

__all__ = ["AtomisticModel", "NeuralNetworkPotential", "NeuralNetworkPotentialNoForces"]


class AtomisticModel(nn.Module):
    """
    Base class for atomistic neural network models.
    """

    def __init__(
        self,
        postprocessors: Optional[List[Transform]] = None,
        do_postprocessing: bool = False,
    ):
        """
        Args:
            postprocessors: Post-processing transforms that may be
                initialized using the `datamodule`, but are not
                applied during training.
            do_postprocessing: If true, post-processing is activated.
        """
        super().__init__()
        self.do_postprocessing = do_postprocessing
        self.postprocessors = nn.ModuleList(postprocessors)
        self.required_derivatives: Optional[List[str]] = None
        self.model_outputs: Optional[List[str]] = None

    def collect_derivatives(self) -> List[str]:
        self.required_derivatives = None
        required_derivatives = set()
        for m in self.modules():
            if (
                hasattr(m, "required_derivatives")
                and m.required_derivatives is not None
            ):
                required_derivatives.update(m.required_derivatives)
        required_derivatives: List[str] = list(required_derivatives)
        self.required_derivatives = required_derivatives

    def collect_outputs(self) -> List[str]:
        self.model_outputs = None
        model_outputs = set()
        for m in self.modules():
            if hasattr(m, "model_outputs") and m.model_outputs is not None:
                model_outputs.update(m.model_outputs)
        model_outputs: List[str] = list(model_outputs)
        self.model_outputs = model_outputs

    def initialize_derivatives(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        #revised, avoid using isinstance and data.to_dict()
        for p in self.required_derivatives:
            if p in data:
                data[p].requires_grad_(True)
        # for p in self.required_derivatives:
        #     if isinstance(data, torch_geometric.Batch): 
        #         if p in data.to_dict().keys():
        #             data[p].requires_grad_(True)
        #     else:
        #         if p in data.keys():
        #             data[p].requires_grad_(True)
        return data

    def initialize_transforms(self, datamodule):
        for module in self.modules():
            if isinstance(module, Transform):
                module.datamodule(datamodule)

    def postprocess(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.do_postprocessing:
            # apply postprocessing
            for pp in self.postprocessors:
                data = pp(data)
        return data

    def extract_outputs(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        results = {k: data[k] for k in self.model_outputs}
        return results


class NeuralNetworkPotential(AtomisticModel):
    """
    A generic neural network potential class that sequentially applies a list of input
    modules, a representation module and a list of output modules.

    This can be flexibly configured for various, e.g. property prediction or potential
    energy sufaces with response properties.
    """

    def __init__(
        self,
        representation: nn.Module,
        input_modules: List[nn.Module] = None,
        output_modules: List[nn.Module] = None,
        #input_dtype_str: str = "float32",
        postprocessors: Optional[List[Transform]] = None,
        do_postprocessing: bool = False,
    ):
        """
        Args:
            representation: The module that builds representation from data.
            input_modules: Modules that are applied before representation, e.g. to
                modify input or add additional tensors for response properties.
            output_modules: Modules that predict output properties from the
                representation.
            postprocessors: Post-processing transforms that may be initialized using the
                `datamodule`, but are not applied during training.
            input_dtype_str: The dtype of real data.
            do_postprocessing: If true, post-processing is activated.
        """
        super().__init__(
            #input_dtype_str=input_dtype_str,
            postprocessors=postprocessors,
            do_postprocessing=do_postprocessing,
        )
        self.representation = representation
        self.input_modules = nn.ModuleList(input_modules)
        self.output_modules = nn.ModuleList(output_modules)

        self.collect_derivatives()
        self.collect_outputs()
        self.register_buffer('dummy_buffer', torch.empty(0)) #added

    def forward(self, 
                data: Dict[str, torch.Tensor], 
                training: bool = False, 
                compute_stress: bool = False, 
                compute_virials: bool = False,
                output_index: Optional[int] = None, # only used for multiple-head output #revised int to Optional[int]
                ) -> Dict[str, torch.Tensor]:
        # initialize derivatives for response properties
        data = self.initialize_derivatives(data)

        if 'stress' in self.model_outputs or 'CACE_stress' in self.model_outputs:
            compute_stress = True
        for m in self.input_modules:
            data = m(data, compute_stress=compute_stress, compute_virials=compute_virials)

        data = self.representation(data)

        # Set requires_grad When Replacing None Values
        # **Added: Replace None values with zero tensors**
        # Determine the device to use
        device = self.dummy_buffer.device
        # Replace None values in data
        new_data: Dict[str, torch.Tensor] = torch.jit.annotate(Dict[str, torch.Tensor], {})
        for key, value in data.items():
            if value is None:
                if key == 'positions':
                    # Determine num_nodes
                    if 'edge_index' in data and data['edge_index'] is not None:
                        edge_index = data['edge_index']
                        assert edge_index is not None
                        num_nodes = edge_index.max().item() + 1
                    else:
                        num_nodes = 1  # Default to 1 if edge_index is not available
                    # Create tensor without 'requires_grad' in constructor
                    tensor = torch.zeros(num_nodes, 3, device=device)
                    tensor.requires_grad_()  # Set requires_grad separately
                    new_data[key] = tensor
                elif key == 'displacement':
                    # Determine num_edges
                    if 'edge_index' in data and data['edge_index'] is not None:
                        edge_index = data['edge_index']
                        assert edge_index is not None
                        num_edges = edge_index.max().item() + 1
                    else:
                        num_edges = 1  # Default to 1 if edge_index is not available
                    # Create tensor without 'requires_grad' in constructor
                    tensor = torch.zeros(num_edges, 3, device=device)
                    tensor.requires_grad_()  # Set requires_grad separately
                    new_data[key] = tensor
                else:
                    # Default zero tensor for other keys
                    new_data[key] = torch.zeros(1, device=device)
            else:
                new_data[key] = value
        # Reassign data to the new dictionary with only Tensor values
        data = new_data

        for key, tensor in data.items():
            assert isinstance(tensor, torch.Tensor), f"{key} is not a Tensor"
            assert tensor is not None, f"{key} is still None"
            assert tensor.numel() > 0, f"{key} is an empty Tensor"
            # Optional: Log tensor shapes
            # print(f"Tensor '{key}': shape={tensor.shape}, device={tensor.device}, requires_grad={tensor.requires_grad}")

        for m in self.output_modules:
            data = m(data, training=training, output_index=output_index)

        # apply postprocessing (if enabled)
        data = self.postprocess(data)

        results = self.extract_outputs(data)

        return results
    
class NeuralNetworkPotentialNoForces(AtomisticModel):
    """
    NeuralNetworkPotential class excluding the Forces module.
    Does not use required_derivatives and model_outputs to maintain TorchScript compatibility.
    """
    def __init__(
        self,
        representation: nn.Module,
        input_modules: Optional[List[nn.Module]] = None,
        output_modules: Optional[List[nn.Module]] = None,
        postprocessors: Optional[List[Transform]] = None,
        do_postprocessing: bool = False,
    ):
        super().__init__(
            postprocessors=postprocessors,
            do_postprocessing=do_postprocessing,
        )
        self.representation = representation
        self.input_modules = nn.ModuleList(input_modules) if input_modules else nn.ModuleList()
        self.output_modules = nn.ModuleList(output_modules) if output_modules else nn.ModuleList()
        self.register_buffer('dummy_buffer', torch.empty(0))  # Added to allow TorchScript to infer the device

        # Set model_outputs to the model_outputs of output_modules
        self.model_outputs = []
        for m in self.output_modules:
            if hasattr(m, 'model_outputs') and m.model_outputs:
                self.model_outputs.extend(m.model_outputs)
        # Remove duplicates
        self.model_outputs = list(set(self.model_outputs))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_stress: bool = False,
        compute_virials: bool = False,
        output_index: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        device = self.dummy_buffer.device

        # Process data through input modules
        for m in self.input_modules:
            data = m(data, compute_stress=compute_stress, compute_virials=compute_virials)
        # Apply representation module
        data = self.representation(data)

        # Separate conditionals to allow TorchScript to clearly infer types
        new_data: Dict[str, torch.Tensor] = {}
        for key, value in data.items():
            if value is None:
                new_data[key] = torch.zeros(1, device=device, dtype=torch.float32)
            else:
                new_data[key] = value
        data = new_data

        # Process data through output modules
        for m in self.output_modules:
            data = m(data, training=training, output_index=output_index)
        # Apply postprocessing
        data = self.postprocess(data)
        # Extract results
        results = self.extract_outputs(data)

        return results