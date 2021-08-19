import os
import pydicom
import pandas as pd
from torch.utils.data import Dataset

assert pydicom.pixel_data_handlers.pylibjpeg_handler.is_available()


class ImageDataset(Dataset):
    """Dataset featuring raw meta data and the pydicom FileDataset of all images

    Usage:
        data = ImageDataset("data/train", "data/train_image_level.csv")
    """
    image_table: pd.DataFrame

    def __init__(self, input_dir: str, image_csv: str) -> None:
        self.image_table = pd.read_csv(image_csv)

    def __len__(self) -> int:
        return len(self.image_table)

    def __getitem__(self, index: int) -> tuple[pydicom.dataset.FileDataset,
                                               pd.core.series.Series]:
        meta = self.image_table.iloc[index]
        image_id = meta["id"].split("_image")[0]
        assert len(image_id) == 12

        # find file
        filename = image_id + ".dcm"
        study_path = os.path.join("data/train", meta["StudyInstanceUID"])
        for sub_dir in os.listdir(study_path):
            image_dir_path = os.path.join(study_path, sub_dir)
            if filename in os.listdir(image_dir_path):
                break
        else:
            raise RuntimeError("404 file not found")
        image_path = os.path.join(image_dir_path, filename)

        dicom = pydicom.read_file(image_path)
        return dicom, meta
