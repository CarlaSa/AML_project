import torch
from torchvision.transforms import Resize
from typeguard import typechecked
from typing import Tuple, Dict

from ..trafo import Trafo, Transformable
from ..rules import numpy_as_3d_tensor
from utils import BoundingBoxes, CanvasTrafoRecorder


@numpy_as_3d_tensor
class Scale(Trafo):
    """
    Rescale an image of any input size to a fixed target size.
    """
    target_size: Tuple[int, int]  # height, width
    torch_transform: Resize

    def __init__(self, target_size: Tuple[int, int]):
        self.target_size = target_size
        self.torch_transform = Resize(target_size)

    @typechecked
    def compute_parameters(self, first_transformand: Transformable,
                           *additional_transformands: Transformable) \
            -> Dict[str, int]:
        return {
            "original_height": first_transformand.shape[-2],
            "original_width": first_transformand.shape[-1]
        }


@Scale.transform.register
def _(self, boxes: BoundingBoxes,
      original_height: int, original_width: int) -> BoundingBoxes:
    """Bounding box scaling"""
    height, width = self.target_size
    boxes[:, (0, 2)] *= width/original_width
    boxes[:, (1, 3)] *= height/original_height
    return boxes


@Scale.transform.register
def _(self, recorder: CanvasTrafoRecorder,
      original_height: int, original_width: int) -> BoundingBoxes:
    """Bounding box scaling"""
    height, width = self.target_size
    recorder.scale(height/original_height, width/original_width)
    return recorder


@Scale.transform.register
def _(self, transformand: torch.Tensor,
      original_width: int, original_height: int) -> torch.Tensor:
    return self.torch_transform(transformand)
