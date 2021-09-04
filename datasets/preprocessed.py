import pydicom
import torch
from torch.utils.data import Dataset
from typeguard import typechecked
from typing import Optional

from .raw_image_dataset import RawImageDataset
from trafo import Trafo, Compose
from trafo.type_mutating import DicomToNDArray, NDArrayTo3dTensor
from trafo.color import Color0ToMax, TruncateGrayValues
from trafo.box_mutating import CropToLungs, CropPadding, Scale, \
    RoundBoundingBoxes
from utils.bounding_boxes import BoundingBoxes

assert pydicom.pixel_data_handlers.pylibjpeg_handler.is_available()


@typechecked
class Preprocessed(Dataset):
    """
    Dataset featuring n bounding boxes (n×4 NumPy array) versus transformed
    image Tensor for each image.

    Usage:
        raw_data = RawImageDataset("data/train", "data/train_image_level.csv")
        data = Preprocessed(raw_data, img_size=(256, 256))
    """
    image_dataset: RawImageDataset
    img_size: tuple[int, int]
    max_bounding_boxes: int
    trafo: Trafo

    label_type: type = BoundingBoxes

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
               Color0ToMax(1),
               RoundBoundingBoxes()
            )
            self.image_dataset = image_dataset
            self.img_size = img_size
            self.max_bounding_boxes = max_bounding_boxes

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, index) -> tuple[torch.Tensor, BoundingBoxes]:
        dicom, meta = self.image_dataset[index]

        img = dicom
        boxes = BoundingBoxes.from_json(meta["boxes"], self.max_bounding_boxes)

        img, boxes = self.trafo(img, boxes)
        return img, boxes
