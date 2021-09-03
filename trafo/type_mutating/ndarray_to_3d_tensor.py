import torch
import numpy as np

from ..trafo import Trafo
from ..rules import preserve_bounding_boxes


@preserve_bounding_boxes
class NDArrayTo3dTensor(Trafo):
    """
    Convert a numpy.ndarray to a 3d torch.Tensor
    """

    pass


@NDArrayTo3dTensor.transform.register
def _(self, array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).reshape((1, *array.shape))
