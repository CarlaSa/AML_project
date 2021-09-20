from sklearn.model_selection import train_test_split
import random
import os
from datetime import datetime
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
from shutil import copyfile
from enum import Enum
from argparse import ArgumentParser

from datasets import LoadDataset, CustomOutput
from datasets.custom_output import image_tensor, bounding_boxes, float_mask
from trafo.randomize.default_augmentation import default_augmentation, \
    default_augmentation_only_values, default_augmentation_only_geometric, \
    default_augmentation_brightness_and_geometric, \
    bounding_boxes_to_tensor_only
from network.unet import Unet
from network.variable_unet import Unet as VariableUnet
from network.Model import OurModel
from network.losses import DiceLoss, BCEandDiceLoss
import torch.nn as nn


class ArgparseEnum(Enum):
    def __str__(self):
        return self.name


class Criterion(ArgparseEnum):
    DICE = DiceLoss()
    BCE = nn.BCELoss().cuda()
    BD = BCEandDiceLoss()
    B2D = BCEandDiceLoss(dice_factor=2)
    B3D = BCEandDiceLoss(dice_factor=3)
    B1p5D = BCEandDiceLoss(dice_factor=1.5)


class Augmentation(ArgparseEnum):
    NA = bounding_boxes_to_tensor_only
    DA = default_augmentation
    DAOG = default_augmentation_only_geometric
    DAOV = default_augmentation_only_values
    DABG = default_augmentation_brightness_and_geometric


def get_args(*args):
    parser = ArgumentParser(description="Train Unet, save weights + results.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--absolute-training-size", type=int, default=5200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--variable-unet", action='store_true')
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--n-initial-block-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--do-batch-norm", action='store_true')
    parser.add_argument("--p-dropout", type=float, default=0)
    parser.add_argument("--adam-regul-factor", type=float, default=0)
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--get-abbrev-only", action='store_true')
    parser.add_argument("--get-path-only", action='store_true')
    parser.add_argument("--get-cuda-device-count-only", action='store_true')
    parser.add_argument("--no-drop-last", action='store_true')
    parser.add_argument("--path", type=str)
    parser.add_argument("criterion", type=Criterion.__getitem__,
                        choices=Criterion)
    parser.add_argument("augmentation", type=Augmentation.__getitem__,
                        choices=Augmentation)
    if len(args) > 0:
        return parser.parse_args(args)
    return parser.parse_args()


def default_args():
    args = get_args(list(Criterion)[0].name, list(Augmentation)[0].name)
    del args.criterion
    del args.augmentation
    return args


def get_abbrev(args):
    if isinstance(args, list):
        args = get_args(*args)
    abbrev = (f"a{args.augmentation.name}_c{args.criterion.name}"
              + f"_b{args.batch_size}_e{args.epochs}")
    if args.do_batch_norm is True:
        abbrev += "_BN"
    default = default_args()
    if args.p_dropout != default.p_dropout:
        abbrev += f"_do{args.p_dropout}"
    if args.variable_unet is True:
        abbrev += "_varUnet"
        if args.n_blocks != default.n_blocks:
            abbrev += f"_blk{args.n_blocks}"
        if args.n_initial_block_channels != default.n_initial_block_channels:
            abbrev += f"_ch{args.n_initial_block_channels}"
    if args.learning_rate != default.learning_rate:
        abbrev += f"_lr{args.learning_rate}"
    if args.adam_regul_factor != default.adam_regul_factor:
        abbrev += f"_wd{args.adam_regul_factor}"
    return abbrev


def get_path(args):
    if args.path is not None:
        return args.path
    path = f"{datetime.now().strftime('%d-%m_%H-%M')}_{get_abbrev(args)}"
    return os.path.join("./_trainings/", path)


def main(*args):
    args = get_args(*args)
    abbrev = get_abbrev(args)  # abbreviation to save meta data etc.
    path = get_path(args)

    if args.get_abbrev_only is True:
        print(abbrev)
        return abbrev
    if args.get_path_only is True:
        print(path)
        return path
    if args.get_cuda_device_count_only is True:
        device_count = torch.cuda.device_count()
        print(device_count)
        return device_count

    if args.do_batch_norm and args.p_dropout > 0:
        print("Warning: Batch Normalisation and Dropout was selected."
              + "Use Batch Normalisation.")

    assert torch.cuda.is_available(), "Missing CUDA"

    if args.cuda_device is not None:
        torch.cuda.set_device(args.cuda_device)

    # Set seeds for reproducibility:
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Get data:
    loaded_data = LoadDataset("_data/preprocessed256_new", image_dtype=float,
                              label_dtype=float)
    dataset_plain = CustomOutput(loaded_data, image_tensor, float_mask)
    dataset_aug = CustomOutput(loaded_data, image_tensor, bounding_boxes,
                               trafo=args.augmentation.value)

    # get good split of dataset -> dividable by batch_size
    l_data = len(dataset_aug)
    indices = list(range(l_data))
    # train_size = (l_data // (args.batch_size * 6)) * args.batch_size * 5
    train_size = args.absolute_training_size
    val_size = l_data - train_size

    print("Training: ", train_size, "Validation: ", val_size)
    train_indices, val_indices = train_test_split(indices, random_state=4,
                                                  train_size=train_size,
                                                  test_size=val_size,
                                                  drop_last=(not
                                                             args.no_drop_last))

    train_set = Subset(dataset_aug, train_indices)
    val_set = Subset(dataset_plain, val_indices)

    dataloader_train = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
    dataloader_val = DataLoader(val_set, batch_size=args.batch_size,
                                shuffle=True, num_workers=0, pin_memory=True)

    if os.path.exists(path):
        print(f"{path} already exists")
    else:
        print(f"Make {path} directory")
        os.makedirs(path)
    copyfile(__file__, os.path.join(path, os.path.basename(__file__)))
    import network.unet
    import network.variable_unet
    for module in (network.unet, network.variable_unet):
        copyfile(module.__file__,
                 os.path.join(path, os.path.basename(module.__file__)))

    if args.variable_unet is True:
        network = VariableUnet(batch_norm=args.do_batch_norm,
                               n_blocks=args.n_blocks,
                               n_initial_block_channels=args.n_initial_block_channels)
    else:
        network = Unet(batch_norm=args.do_batch_norm, p_dropout=args.p_dropout)

    Model = OurModel(name="unet", network=network,
                     criterion=args.criterion.value, path_dir=path,
                     lr=args.learning_rate, batch_size=args.batch_size,
                     verbose=True, segmentation=True,
                     data_trafo=dataset_aug.trafo,
                     adam_regul_factor=args.adam_regul_factor
                     )

    # save a json file which indicates what parameters are used for training
    Model.save_configuration()
    Model.train(args.epochs, dataloader_train, validate=True,
                dataloader_val=dataloader_val, save_observables=True)


if __name__ == '__main__':
    main()
