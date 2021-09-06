from datasets import LoadDataset, CustomOutput, Knit
from datasets.custom_output import index, image_csv_index, image_csv_meta, \
    image_id, image_tensor, study_label, \
    bounding_boxes, mask, masked_image_tensor

loaded_data = LoadDataset("_data/preprocessed256", image_dtype=float,
                          label_dtype=float)
knit_data = Knit(loaded_data,
                 image_csv="_data/train_image_level.csv",
                 study_csv="_data/train_study_level.csv")

data = CustomOutput(knit_data,
                    masked_image_tensor, study_label, image_id)
data[42]
