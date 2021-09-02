import os
import pydicom
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision
import PIL.Image
from warnings import warn

from utils.bounding_boxes import bounding_boxes_array

assert pydicom.pixel_data_handlers.pylibjpeg_handler.is_available()


class ImageDataset(Dataset):
    """
    Dataset featuring raw meta data (as Pandas Series) versus the pydicom
    FileDataset for each image.

    Usage:
        data = ImageDataset("data/train", "data/train_image_level.csv")
    """
    image_table: pd.DataFrame
    input_dir: str

    def __init__(self, input_dir: str, image_csv: str) -> None:
        self.image_table = pd.read_csv(image_csv)
        self.input_dir = input_dir

    def __len__(self) -> int:
        return len(self.image_table)

    def __getitem__(self, index: int) -> tuple[pydicom.dataset.FileDataset,
                                               pd.core.series.Series]:
        dicom: pydicom.dataset.FileDataset
        meta: pd.core.series.Series

        meta = self.image_table.iloc[index]
        image_id = meta["id"].split("_image")[0]
        assert len(image_id) == 12, "Corrupt CSV data. Is this the right file?"

        # find file
        filename = image_id + ".dcm"
        study_path = os.path.join(self.input_dir, meta["StudyInstanceUID"])
        for sub_dir in os.listdir(study_path):
            image_dir_path = os.path.join(study_path, sub_dir)
            if filename in os.listdir(image_dir_path):
                break
        else:
            raise RuntimeError("404 file not found")
        image_path = os.path.join(image_dir_path, filename)

        dicom = pydicom.read_file(image_path)
        return dicom, meta


class UniformImageDataset(Dataset):
    """
    DEPRECATED.

    Dataset featuring n bounding boxes (n×4 NumPy array) versus transformed
    PIL image for each image.

    Usage:
        image_data = ImageDataset("data/train", "data/train_image_level.csv")
        data = UniformImageDataset(image_data, img_size=(1024, 1024))
    """
    image_dataset: ImageDataset
    img_size: tuple[int, int]
    max_bounding_boxes: int

    def __init__(self, image_dataset: ImageDataset,
                 img_size: tuple[int, int] = (1024, 1024),
                 max_bounding_boxes: int = 8):
        warn("UniformImageDataset is DEPRECATED and should not be used.",
             DeprecationWarning, stacklevel=2)
        self.image_dataset = image_dataset
        self.img_size = img_size
        self.max_bounding_boxes = max_bounding_boxes

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, index) -> tuple:
        dicom, meta = self.image_dataset[index]

        img_array = dicom.pixel_array
        orig_height, orig_width = img_array.shape
        img = pil_image_from_array(img_array)
        img = torchvision.transforms.Resize(self.img_size)(img)
        img = torchvision.transforms.ToTensor()(img)

        boxes = bounding_boxes_array(meta["boxes"], self.max_bounding_boxes)

        # resize box:
        # assuming that torchvision.transforms.Resize does rescale and not crop
        width, height = self.img_size
        boxes[:, (0, 2)] *= width/orig_width  # x, width
        boxes[:, (1, 3)] *= height/orig_height  # y, height

        # TODO: RandomHorizontalFlip

        return img, boxes


def pil_image_from_array(array: np.array) -> PIL.Image:
    """
    Convert pixel_array to PIL image (change the pixel value range to [0,255]).

    Mainly taken from
    https://stackoverflow.com/questions/42650233/how-to-access-rgb-pixel-arrays-from-dicom-files-using-pydicom

    Args:
        array (np.array): NumPy array to be converted

    Returns:
        PIL.Image: the desired image converted
    """
    img = array.astype(float)
    img = (np.maximum(img, 0) / img.max()) * 255.0
    img = np.uint8(img)
    return PIL.Image.fromarray(img)
