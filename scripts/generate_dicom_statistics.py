import json
import pandas as pd
from tqdm import tqdm
from datasets.raw_image_dataset import RawImageDataset


def main():
    data = RawImageDataset("data/train", "data/train_image_level.csv")
    study_table = pd.read_csv("data/train_study_level.csv")
    table = pd.DataFrame()
    keywords = {}
    exclude = {"PixelData"}
    for dicom, meta in tqdm(data):
        assert dicom.StudyInstanceUID == meta.StudyInstanceUID
        study_label = (study_table.loc[study_table["id"]
                                       == f"{meta.StudyInstanceUID}_study"]
                       == 1).idxmax(axis=1).values[0]
        dicom.__repr__()
        elements = list(dicom.elements())
        array = dicom.pixel_array
        table = table.append({**dict(meta),
                              "n_boxes": len(meta.label.split("opacity")) - 1,
                              "study_label": study_label,
                              "pixel_min": array.min(),
                              "pixel_mean": array.mean(),
                              "pixel_max": array.max(),
                              **{el.keyword: el.value for el in elements
                                 if el.keyword not in exclude}},
                             ignore_index=True)
        keywords = {**keywords, **{el.keyword: el.name for el in elements}}

    table.to_csv("artifacts/dicom_meta.csv")
    with open("artifacts/dicom_keywords.json", "w") as f:
        json.dump(keywords, f)


if __name__ == '__main__':
    main()
