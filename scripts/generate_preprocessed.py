from datasets import RawImageDataset, Preprocessed, save_dataset, LoadDataset
# from matplotlib import pyplot as plt


def main() -> None:
    raw_data = RawImageDataset("data/train", "data/train_image_level.csv")
    image_filenames = [id.replace("_image", ".png")
                       for id in raw_data.image_table["id"]]
    preprocessed_data = Preprocessed(raw_data, (256, 256))

    directory = "_data/preprocessed256pc"
    # plt.imshow(preprocessed_data[0][0][0])
    save_dataset(preprocessed_data, directory, image_filenames)

    # data = LoadDataset(directory, image_dtype=float, label_dtype=float)


if __name__ == '__main__':
    main()
