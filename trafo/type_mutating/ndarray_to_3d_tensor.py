import torch
import numpy as np

from ..trafo import Trafo
from ..rules import preserve
from utils import BoundingBoxes, CanvasTrafoRecorder


@preserve(BoundingBoxes, CanvasTrafoRecorder)
class NDArrayTo3dTensor(Trafo):
    """
    Convert a numpy.ndarray to a 3d torch.Tensor.

    Return torch.Tensor objects unchanged.
    """

    pass


@NDArrayTo3dTensor.transform.register
def _(self, array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).reshape((1, *array.shape))


@NDArrayTo3dTensor.transform.register
def _(self, tensor: torch.Tensor) -> torch.Tensor:
    return tensor
