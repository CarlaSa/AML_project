import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm
import csv

from typing import Iterable, Sequence
from itertools import count


def save_dataset(dataset: Dataset, directory: str,
                 image_filenames: Iterable[str] = (f"{i:d}.png"
                                                   for i in count())) -> None:
    """
    Save an existing image dataset into a directory.

    This function reads all the data from a dataset and saves each image into
    a single file (subfolder "images/") and stores the file names and labels if __name__ == '__main__':
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

    if not os.path.exists(directory):
        os.makedirs(directory)
    image_dir = os.path.join(directory, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for (image, label), filename in zip(tqdm(dataset), image_filenames):
        torchvision.utils.save_image(image, os.path.join(image_dir, filename))
        with open(os.path.join(directory, "labels.csv"), "a") as f:
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

    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.table = pd.read_csv(os.path.join(input_dir, "labels.csv"),
                                 header=None)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.ndarray]:
        meta = self.table.iloc[index]
        filename = meta[0]
        label = self.parse_label(meta[1:])
        image = self.load_image(os.path.join(self.input_dir, "images",
                                             filename))
        return image, label

    @staticmethod
    def load_image(path: str) -> torch.Tensor:
        return torchvision.io.read_image(path,
                                         torchvision.io.ImageReadMode.GRAY)

    @staticmethod
    def parse_label(label: Sequence) -> np.ndarray:
        n_rows = len(label)//4
        array = np.array(label)
        return array.reshape((n_rows, 4))
