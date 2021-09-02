from .trafo import Trafo, Transformable
from .composed_trafo import ComposedTrafo

# Type mutating transformations:
from .dicom_to_ndarray import DicomToNDArray

# Bounding box mutating transformations:
# from .scale import Scale
# TODO: implement
# from .crop_to_lungs import CropToLungs
# from .crop_padding import CropPadding

# Bounding box preserving transformations:
# from .to_8_bit_color import To8BitColor
# from .truncate_gray_values import TruncateGrayValues
