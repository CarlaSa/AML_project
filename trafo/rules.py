import torch
import numpy as np

from utils.bounding_boxes import BoundingBoxes
from .trafo import TrafoMeta


def preserve_bounding_boxes(cls: TrafoMeta) -> TrafoMeta:
    @cls.transform.register
    def _(self, boxes: BoundingBoxes, **parameters) -> BoundingBoxes:
        return boxes
    return cls


def preserve_ndarray(cls: TrafoMeta) -> TrafoMeta:
    @cls.transform.register
    def _(self, image: np.ndarray, **parameters) -> np.ndarray:
        return image
    return cls


def preserve_tensor(cls: TrafoMeta) -> TrafoMeta:
    @cls.transform.register
    def _(self, image: torch.Tensor, **parameters) -> torch.Tensor:
        return image
    return cls


def preserve_image(cls: TrafoMeta) -> TrafoMeta:
    @cls.transform.register
    def _(self, image: np.ndarray, **parameters) -> np.ndarray:
        return image

    @cls.transform.register
    def _(self, image: torch.Tensor, **parameters) -> torch.Tensor:
        return image
    return cls


def numpy_as_3d_tensor(cls: TrafoMeta) -> TrafoMeta:
    @cls.transform.register
    def _(self, transformand: np.ndarray, **parameters) -> np.ndarray:
        tensor = torch.from_numpy(transformand).reshape(
            (1, *transformand.shape))
        tensor = self.transform(tensor, **parameters)
        array = tensor[0].numpy()
        return array
    return cls
