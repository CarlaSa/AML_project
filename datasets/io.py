import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm
import csv
from warnings import warn
from typing import Iterable, Sequence, Optional
from itertools import count

from utils.bounding_boxes import BoundingBoxes


def save_dataset(dataset: Dataset, directory: str,
                 image_filenames: Iterable[str] = (f"{i:d}.png"
                                                   for i in count())) -> None:
    """
    Save an existing image dataset into a directory.

    This function reads all the data from a dataset and saves each image into
    a single file (subfolder "images/") and stores the file names and labels in
    a common CSV file ("labels.csv").

    The image file type is determined by the file name of an image.

    Args:
        dataset (Dataset): The dataset to save.
        directory (str): Path to the directory to save the dataset to. If this
            directory does not exist, it will be created. Any existing contents
            will be overwritten but not cleared first.
        image_filenames (Iterable[str]): Names of the files of the images to be
            saved as in the same order as the dataset. Note that the image type
            is inferred from the filename suffix.
            This argument is optional; the default is 0.png, 1.png, ...
    """
    image: torch.Tensor
    label: np.ndarray  # 4×n
    filename: str  # of the image, including suffix
    csv_path: str = os.path.join(directory, "labels.csv")
    image_dir: str = os.path.join(directory, "images")

    if not os.path.exists(directory):
        warn(f"Creating {directory}")
        os.makedirs(directory)
    if not os.path.exists(image_dir):
        warn(f"Creating {image_dir}")
        os.makedirs(image_dir)
    if os.path.exists(csv_path):
        warn(f"Removing old CSV file {csv_path}.")
        os.remove(csv_path)

    for (image, label), filename in zip(tqdm(dataset), image_filenames):
        torchvision.utils.save_image(image, os.path.join(image_dir, filename))
        if label.dtype is np.dtype("bool"):
            label = np.uint(label)
        with open(csv_path, "a") as f:
            csv.writer(f).writerow([filename] + list(label.reshape(-1)))


class LoadDataset(Dataset):
    """A dataset loaded from a directory created using save_dataset.

    Args:
        input_dir (str): Path to the directory to load the dataset from. Images
            are expected in the subdirectory "images/", and a "labels.csv"
            featuring the image filenames in the first column and the labels
            in the remaining columns (without any header rows) is required.
    """
    table: pd.DataFrame
    input_dir: str
    image_dtype: torch.dtype
    label_dtype: np.dtype
    image_ids: Optional[list[str]]
    label_type: Optional[type]

    def __init__(self, input_dir: str, image_dtype: torch.dtype,
                 label_dtype: Optional[np.dtype] = None,
                 filename_is_id: bool = True):
        self.input_dir = input_dir
        self.table = pd.read_csv(os.path.join(input_dir, "labels.csv"),
                                 header=None)
        self.image_dtype = image_dtype
        self.label_dtype = label_dtype
        if filename_is_id:
            self.image_ids = [fn.split(".")[0] for fn in self.table[0]]
        else:
            self.image_ids = None
        self.label_type = type(self.parse_label(self.table.iloc[0][1:],
                                                self.label_dtype))

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.ndarray]:
        meta = self.table.iloc[index]
        filename = meta[0]
        label = self.parse_label(meta[1:], self.label_dtype)
        if type(label) is not self.label_type:
            raise TypeError(f"Varying label types: Found {type(label)}, "
                            + f"expected {self.label_type()}.")
        image = self.load_image(os.path.join(self.input_dir, "images",
                                             filename))
        image = image.to(self.image_dtype)
        return image, label

    def load_image(self, path: str) -> torch.Tensor:
        return torchvision.io.read_image(path,
                                         torchvision.io.ImageReadMode.GRAY)

    @staticmethod
    def parse_label(label: Sequence, dtype: Optional[np.dtype] = None) \
            -> np.ndarray:
        n_rows = len(label)//4
        if dtype is None:
            array = np.array(label)
        else:
            array = np.array(label, dtype=dtype)
        if n_rows == 1 and len(label) == 4:  # only 1-dimensional
            return array
        return BoundingBoxes.from_array(array.reshape((n_rows, 4)))
