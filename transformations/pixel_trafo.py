import numpy as np
from deprecation import deprecated


@deprecated()
def scale8bit(img: np.array) -> np.array:
    """
    Scale the pixel values of an image to the range [0, 255]
    """
    img -= img.min()
    img = (np.maximum(img, 0) / img.max()) * 255.0
    return np.uint8(img)
