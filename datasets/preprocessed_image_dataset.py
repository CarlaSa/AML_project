import pydicom
import torch
from torch.utils.data import Dataset
from typeguard import typechecked
from typing import Optional

from .raw_image_dataset import RawImageDataset
from trafo import Trafo, Compose
from trafo.type_mutating import DicomToNDArray, NDArrayTo3dTensor
from trafo.color import Color0ToMax, To8BitColor, TruncateGrayValues
from trafo.box_mutating import CropToLungs, CropPadding, Scale
from utils.bounding_boxes import bounding_boxes_array, BoundingBoxes

assert pydicom.pixel_data_handlers.pylibjpeg_handler.is_available()


@typechecked
class PreprocessedImageDataset(Dataset):
    """
    Dataset featuring n bounding boxes (n×4 NumPy array) versus transformed
    image Tensor for each image.

    Usage:
        image_data = ImageDataset("data/train", "data/train_image_level.csv")
        data = UniformImageDataset(image_data, img_size=(1024, 1024))
    """
    image_dataset: RawImageDataset
    img_size: tuple[int, int]
    max_bounding_boxes: int
    trafo: Trafo

    def __init__(self, image_dataset: RawImageDataset,
                 img_size: tuple[int, int] = (1024, 1024),
                 max_bounding_boxes: int = 8,
                 trafo: Optional[Trafo] = None):
        if trafo is None:
            self.trafo = Compose(
               DicomToNDArray(),
               TruncateGrayValues(),
               Color0ToMax(255),
               CropToLungs(),
               # CropPadding(),
               NDArrayTo3dTensor(),
               Scale(img_size),
               Color0ToMax(1)
            )
            self.image_dataset = image_dataset
            self.img_size = img_size
            self.max_bounding_boxes = max_bounding_boxes

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, index) -> tuple[torch.Tensor, BoundingBoxes]:
        dicom, meta = self.image_dataset[index]

        img = dicom
        boxes = bounding_boxes_array(
            meta["boxes"], self.max_bounding_boxes)

        img, boxes = self.trafo(img, boxes)

        return img, boxes
