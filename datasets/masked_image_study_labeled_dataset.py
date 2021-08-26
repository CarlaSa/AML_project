import os
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision

from datasets.image_dataset import pil_image_from_array, \
    bounding_boxes_array, bounding_boxes_mask
from datasets.study_dataset import LABEL_KEYS

assert pydicom.pixel_data_handlers.pylibjpeg_handler.is_available()

BOX_ATTRIBUTES = ["x", "y", "width", "height"]


class MaskedImageStudyLabeledDataset(Dataset):
    """
    Dataset featuring study labels versus uniform Tensor for each image.

    Args:
        input_dir (str): path of directory containing image data,
            e.g. "data/train"
        image_csv (str): path to the image level csv file,
            e.g. "data/train_image_level.csv"
        study_csv (str): path to the study level csv file,
            e.g. "data/train_study_level.csv"
        img_size (tuple[int, int]): resize images to this img_size
        fix_monochrome (bool): if MONOCHROME1 images should be inverted first
        max_bounding_boxes (int): maximum number of bounding boxes
    """
    image_table: pd.DataFrame
    study_table: pd.DataFrame
    input_dir: str
    img_size: tuple[int, int]
    fix_monochrome: bool
    max_bounding_boxes: int

    def __init__(self, input_dir: str, image_csv: str, study_csv: str,
                 img_size: tuple[int, int] = (1024, 1024),
                 fix_monochrome: bool = True,
                 max_bounding_boxes: int = 8,
                 transforms=None) -> None:
        self.image_table = pd.read_csv(image_csv)
        self.study_table = pd.read_csv(study_csv)
        self.input_dir = input_dir
        self.img_size = img_size
        self.fix_monochrome = fix_monochrome
        self.max_bounding_boxes = max_bounding_boxes

        if transforms is None:
            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(img_size),
                torchvision.transforms.ToTensor(),
            ])
        else:
            self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_table)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.array]:
        img: torch.Tensor
        label: np.array

        meta = self.image_table.iloc[index]
        image_id = meta["id"].split("_image")[0]
        assert len(image_id) == 12, "Corrupt CSV data. Is this the right file?"

        study_id = meta['StudyInstanceUID']
        study_label = self.study_table.loc[self.study_table["id"]
                                           == f"{study_id}_study"]
        assert len(study_label) == 1, \
            f"Missing or ambiguous study: {study_id}"
        label = np.array(study_label.iloc[0][LABEL_KEYS], dtype=bool)

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
        img_array = dicom.pixel_array
        if self.fix_monochrome and dicom.PhotometricInterpretation \
           == "MONOCHROME1":
            img_array = np.amax(img_array) - img_array

        orig_height, orig_width = img_array.shape
        img = pil_image_from_array(img_array)
        #img = torchvision.transforms.Resize(self.img_size)(img)
        #img = torchvision.transforms.ToTensor()(img)
        img = self.transforms(img)

        boxes = bounding_boxes_array(meta["boxes"], self.max_bounding_boxes)

        # resize box:
        # assuming that torchvision.transforms.Resize does rescale and not crop
        width, height = self.img_size
        boxes[:, (0, 2)] *= width/orig_width  # x, width
        boxes[:, (1, 3)] *= height/orig_height  # y, height

        mask = bounding_boxes_mask(boxes, self.img_size)
        img[0][~torch.from_numpy(mask).bool()] = 0

        return img, label
