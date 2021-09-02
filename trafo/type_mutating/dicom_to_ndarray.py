import numpy as np
import pydicom.dataset

from ..trafo import Trafo
from ..rules import preserve_bounding_box


@preserve_bounding_box
class DicomToNDArray(Trafo):
    """
    Rescale an image of any input size to a fixed target size.
    """
    fix_monochrome: bool

    def __init__(self, fix_monochrome: bool = True):
        self.fix_monochrome = fix_monochrome


@DicomToNDArray.transform.register
def _(self, dicom: pydicom.dataset.FileDataset) -> np.ndarray:
    image = dicom.pixel_array
    if self.fix_monochrome and dicom.PhotometricInterpretation \
       == "MONOCHROME1":
        image = np.amax(image) - image
    return image
