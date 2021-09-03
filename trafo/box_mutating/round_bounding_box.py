import math

from ..trafo import Trafo

@preserve_image
class RoundBoundingBox(Trafo):
    """
    Round floats to integers
    """
    pass

@Color0ToMax.transform.register
def _(self, boxes: np.ndarray) -> np.ndarray:
    boxes[:, :2] = np.floor(boxes[:, :2])
    boxes[:, 2:] = np.ceil(boxes[:, 2:])
    return boxes
