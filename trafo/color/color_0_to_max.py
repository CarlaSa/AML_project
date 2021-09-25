import torch
import numpy as np
from dataclasses import dataclass

from ..trafo import Trafo
from ..rules import preserve
from utils import BoundingBoxes, CanvasTrafoRecorder


@preserve(BoundingBoxes, CanvasTrafoRecorder)
@dataclass
class Color0ToMax(Trafo):
    """Scale the pixel values of an image to the range [0, 255]."""

    max: float


@Color0ToMax.transform.register
def _(self, image: np.ndarray) -> np.ndarray:
    image -= image.min()
    image = (np.maximum(image, 0) / image.max()) * self.max
    return image


@Color0ToMax.transform.register
def _(self, image: torch.Tensor) -> torch.Tensor:
    image -= image.min()
    image = (np.maximum(image, 0) / image.max()) * self.max
    return image
