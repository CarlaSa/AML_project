import os
import torch
import random
import network.full_model
import network.unet
import network.variable_unet
import numpy as np
from shutil import copyfile
from enum import Enum
from datetime import datetime
from argparse import ArgumentParser, Namespace

from trafo.randomize.default_augmentation import default_augmentation, \
    default_augmentation_only_values, default_augmentation_only_geometric, \
    default_augmentation_brightness_and_geometric, \
    bounding_boxes_to_tensor_only


class ArgparseEnum(Enum):
    def __str__(self):
        return self.name


class Augmentation(ArgparseEnum):
    NA = bounding_boxes_to_tensor_only
    DA = default_augmentation
    DAOG = default_augmentation_only_geometric
    DAOV = default_augmentation_only_values
    DABG = default_augmentation_brightness_and_geometric


class CLITraining:
    args: Namespace
    abbrev: str
    path: str

    def __init__(self, args: Namespace, abbrev: str):
        self.args = args
        self.abbrev = abbrev
        self.path = self.get_path(args, abbrev)

    @staticmethod
    def get_path(args: Namespace, abbrev: str) -> str:
        if args.path is not None:
            return args.path
        else:
            path = f"{datetime.now().strftime('%d-%m_%H-%M')}_{abbrev}"
            return os.path.join("./_trainings/", path)

    def main(self):
        if self.args.get_abbrev_only is True:
            print(self.abbrev)
            return self.abbrev
        if self.args.get_path_only is True:
            print(self.path)
            return self.path
        if self.args.get_cuda_device_count_only is True:
            device_count = torch.cuda.device_count()
            print(device_count)
            return device_count
        return self.run()

    def run(self):
        assert torch.cuda.is_available(), "Missing CUDA"

        if self.args.cuda_device is not None:
            torch.cuda.set_device(self.args.cuda_device)

        if os.path.exists(self.path):
            print(f"{self.path} already exists")
            if self.args.path is not None:
                raise FileExistsError(self.path)
        else:
            print(f"Make {self.path} directory")
            os.makedirs(self.path)
        copyfile(__file__, os.path.join(self.path, os.path.basename(__file__)))

        for module in ("unet", "variable_unet", "full_model"):
            file = getattr(network, module).__file__
            copyfile(file, os.path.join(self.path, os.path.basename(file)))

        # Set seeds for reproducibility:
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)


class TrainingCLI:
    parser: ArgumentParser
    training_class: type = CLITraining

    def __init__(self, parser: ArgumentParser = ArgumentParser()):
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--absolute-training-size", type=int, default=5200)
        parser.add_argument("--batch-size", type=int, default=16)
        parser.add_argument("--learning-rate", type=float, default=0.001)
        parser.add_argument("--do-batch-norm", action='store_true')
        parser.add_argument("--adam-regul-factor", type=float, default=0)
        parser.add_argument("--cuda-device", type=int, default=None)
        parser.add_argument("--get-abbrev-only", action='store_true')
        parser.add_argument("--get-path-only", action='store_true')
        parser.add_argument("--get-cuda-device-count-only",
                            action='store_true')
        parser.add_argument("--no-drop-last", action='store_true')
        parser.add_argument("--use-lr-scheduler", action='store_true')
        parser.add_argument("--lr-sch-patience", type=int, default=10)
        parser.add_argument("--path", type=str)
        self.parser = parser

    def get_args(self, *args: str) -> Namespace:
        if len(args) > 0:
            return self.parser.parse_args(args)
        return self.parser.parse_args()

    def default_args(self) -> Namespace:
        return self.get_args()

    def get_abbrev(self, args: Namespace) -> str:
        abbrev = (f"a{args.augmentation.name}_c{args.criterion.name}"
                  + f"_b{args.batch_size}_e{args.epochs}")
        if args.do_batch_norm is True:
            abbrev += "_BN"
        default = self.default_args()
        if args.learning_rate != default.learning_rate:
            abbrev += f"_lr{args.learning_rate}"
        if args.use_lr_scheduler is True:
            abbrev += f"_lrsp{args.lr_sch_patience}"
        if args.adam_regul_factor != default.adam_regul_factor:
            abbrev += f"_wd{args.adam_regul_factor}"
        return abbrev

    def get_training(self, *args: str) -> CLITraining:
        _args = self.get_args(*args)
        return self.training_class(_args, self.get_abbrev(_args))
