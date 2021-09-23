import os
import torch
from torchvision import io
from typing import List, Tuple

from torch.utils.data import Dataset


class TestData(Dataset):
    """Dataset featuring ids versus image tensor."""

    image_dir: str
    filenames: List[str]

    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.filenames = [fn for fn in os.listdir(image_dir)
                          if fn.endswith(".png")]

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index) -> Tuple[torch.Tensor, str, str]:
        filename = self.filenames[index]
        path = os.path.join(self.image_dir, filename)
        study_id, image_id = filename.split(".")[0].split("_")
        assert len(study_id) == 12 and len(image_id) == 12
        tensor = io.read_image(path, io.ImageReadMode.GRAY).to(float)/255
        return tensor, study_id, image_id
