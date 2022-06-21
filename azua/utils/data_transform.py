from typing import Any, Callable, Dict, List, Optional

import torch
import torchvision
from torch.utils.data import TensorDataset


def _instantiate_torchvision_transform(class_name: str, **kwargs: Any):
    return getattr(torchvision.transforms, class_name)(**kwargs)


def transform_from_configs(transform_cfgs: List[Dict[str, Any]]):
    transforms = [_instantiate_torchvision_transform(**cfg) for cfg in transform_cfgs]
    return torchvision.transforms.Compose(transforms)


class TransformableTensorDataset(TensorDataset):
    """Adds support for torchvision transforms to pytorch's TensorDataset class. Takes a list of tensors and transforms
    and maps the transforms over the tensors when retrieving an item."""

    def __init__(
        self,
        tensors: List[torch.Tensor],
        transforms: List[Optional[Callable[[torch.Tensor], torch.Tensor]]],
    ):
        super().__init__(*tensors)
        if len(tensors) != len(transforms):
            raise ValueError("Must provide a transform (which may be None) for every data tensor.")
        self.transforms = transforms

    def __getitem__(self, index):
        data = super().__getitem__(index)
        data = tuple(tf(d) if tf is not None else d for d, tf in zip(data, self.transforms))
        return data


class TransformableFirstTensorDataset(TransformableTensorDataset):
    """Convenience class that takes a transform for only the first tensor."""

    def __init__(self, *tensors: torch.Tensor, transform: Callable[[torch.Tensor], torch.Tensor]):
        transforms = [transform] + [None] * (len(tensors) - 1)  # type: ignore
        super().__init__(tensors, transforms)  # type: ignore
