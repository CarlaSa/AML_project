import numpy as np

from ..trafo import Trafo
from ..rules import preserve
from utils import BoundingBoxes, CanvasTrafoRecorder


@preserve(BoundingBoxes, CanvasTrafoRecorder)
class TruncateGrayValues(Trafo):
    """
    Truncate gray values to a maximum and a minimum.
    """
    factor_max: float
    factor_min: float

    def __init__(self, factor_max: float = 0.5, factor_min: float = 0.5) \
            -> None:
        self.factor_max = factor_max
        self.factor_min = factor_min


@TruncateGrayValues.transform.register
def _(self, image: np.ndarray) -> np.ndarray:
    max_val = np.mean(image) + self.factor_max * np.max(image)
    min_val = np.mean(image) - self.factor_min * np.max(image)
    image[image > max_val] = max_val
    image[image < min_val] = min_val
    return image
