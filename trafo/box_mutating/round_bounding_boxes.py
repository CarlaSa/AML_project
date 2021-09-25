import torch
import numpy as np

from ..trafo import Trafo
from ..rules import preserve
from utils import BoundingBoxes, CanvasTrafoRecorder


@preserve(np.ndarray, torch.Tensor, CanvasTrafoRecorder)
class RoundBoundingBoxes(Trafo):
    """
    Round floats to integers
    """
    pass


@RoundBoundingBoxes.transform.register
def _(self, boxes: BoundingBoxes) -> BoundingBoxes:
    boxes[:, :2] = np.floor(boxes[:, :2])
    boxes[:, 2:] = np.ceil(boxes[:, 2:])
    return boxes.astype('int')
