import torch
import pandas as pd
from torch.utils.data import Dataset
from typeguard import typechecked
from typing import Optional, Tuple

from .raw_image_dataset import RawImageDataset
from trafo import Trafo, Compose
from trafo.type_mutating import NDArrayTo3dTensor
from trafo.type_mutating.dicom_to_ndarray import DicomToNDArray
from trafo.color import Color0ToMax, TruncateGrayValues
from trafo.box_mutating import CropToLungs, CropPadding, Scale, \
    RoundBoundingBoxes
from utils.bounding_boxes import BoundingBoxes


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
    image_table: pd.DataFrame
    img_size: Tuple[int, int]
    max_bounding_boxes: int
    trafo: Trafo

    label_type: type = BoundingBoxes

    def __init__(self, image_dataset: RawImageDataset,
                 img_size: Tuple[int, int] = (1024, 1024),
                 max_bounding_boxes: int = 8,
                 trafo: Optional[Trafo] = None):
        if trafo is None:
            self.trafo = Compose(
               DicomToNDArray(),
               TruncateGrayValues(),
               Color0ToMax(255),
               CropToLungs(),
               CropPadding(),
               NDArrayTo3dTensor(),
               Scale(img_size),
               Color0ToMax(1),
               RoundBoundingBoxes()
            )
            self.image_dataset = image_dataset
            self.image_table = image_dataset.image_table
            self.img_size = img_size
            self.max_bounding_boxes = max_bounding_boxes

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, index) -> Tuple[torch.Tensor, BoundingBoxes]:
        dicom, meta = self.image_dataset[index]

        img = dicom
        boxes = BoundingBoxes.from_json(meta["boxes"], self.max_bounding_boxes)

        img, boxes = self.trafo(img, boxes)
        return img, boxes
