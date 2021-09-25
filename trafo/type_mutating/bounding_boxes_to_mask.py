import torch
import numpy as np
from typing import Dict
from typeguard import typechecked

from ..trafo import Trafo, Transformable
from ..rules import preserve
from utils import BoundingBoxes, CanvasTrafoRecorder


@preserve(torch.Tensor, CanvasTrafoRecorder)
class BoundingBoxesToMask(Trafo):
    """
    Convert a numpy.ndarray to a 3d torch.Tensor.

    Return torch.Tensor objects unchanged.
    """
    @typechecked
    def compute_parameters(self, image: torch.Tensor,
                           *additional_transformands: Transformable) \
            -> Dict[str, int]:
        return {
            "original_height": image.shape[-2],
            "original_width": image.shape[-1]
        }


@BoundingBoxesToMask.transform.register
def _(self, boxes: BoundingBoxes,
      original_height: float, original_width: float) -> torch.Tensor:
    return boxes.get_float_mask((original_height, original_width))
