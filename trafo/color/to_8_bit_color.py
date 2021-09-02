import numpy as np

from ..trafo import Trafo
from ..rules import preserve_bounding_boxes


@preserve_bounding_boxes
class To8BitColor(Trafo):
    """
    Scale the pixel values of an image to the range [0, 255]
    """
    pass


@To8BitColor.transform.register
def _(self, image: np.ndarray) -> np.ndarray:
    image -= image.min()
    image = (np.maximum(image, 0) / image.max()) * 255
    return np.uint8(image)
