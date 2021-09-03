import numpy as np

from ..trafo import Trafo
from ..rules import preserve_image


@preserve_image
class RoundBoundingBoxes(Trafo):
    """
    Round floats to integers
    """
    pass


@RoundBoundingBoxes.transform.register
def _(self, boxes: np.ndarray) -> np.ndarray:
    boxes[:, :2] = np.floor(boxes[:, :2])
    boxes[:, 2:] = np.ceil(boxes[:, 2:])
    return boxes.astype('int')
