from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from argparse import ArgumentParser, Namespace

from datasets import LoadDataset, CustomOutput
from datasets.custom_output import image_tensor, bounding_boxes, float_mask

from network.unet import Unet
from network.variable_unet import Unet as VariableUnet
from network.Model import OurModel
from network.losses import DiceLoss, BCEandDiceLoss

from .common_train import TrainingCLI, Augmentation, ArgparseEnum, CLITraining


class Criterion(ArgparseEnum):
    DICE = DiceLoss()
    BCE = nn.BCELoss().cuda()
    BD = BCEandDiceLoss()
    B2D = BCEandDiceLoss(dice_factor=2)
    B3D = BCEandDiceLoss(dice_factor=3)
    B4D = BCEandDiceLoss(dice_factor=4)
    B1p5D = BCEandDiceLoss(dice_factor=1.5)


class UnetCLITraining(CLITraining):
    def run(self):
        super().run()
        if self.args.do_batch_norm and self.args.p_dropout > 0:
            print("Warning: Batch Normalisation and Dropout was selected."
                  + "Use Batch Normalisation.")

        # Get data:
        loaded_data = LoadDataset("_data/preprocessed256_new", image_dtype=float,
                                  label_dtype=float)
        dataset_plain = CustomOutput(loaded_data, image_tensor, float_mask)
        dataset_aug = CustomOutput(loaded_data, image_tensor, bounding_boxes,
                                   trafo=self.args.augmentation.value)

        # get good split of dataset -> dividable by batch_size
        l_data = len(dataset_aug)
        indices = list(range(l_data))
        # train_size = (l_data // (self.args.batch_size * 6)) * self.args.batch_size * 5
        train_size = self.args.absolute_training_size
        val_size = l_data - train_size

        print("Training: ", train_size, "Validation: ", val_size)
        train_indices, val_indices = train_test_split(indices, random_state=4,
                                                      train_size=train_size,
                                                      test_size=val_size,
                                                      )

        train_set = Subset(dataset_aug, train_indices)
        val_set = Subset(dataset_plain, val_indices)

        dataloader_train = DataLoader(train_set,
                                      batch_size=self.args.batch_size,
                                      shuffle=True, num_workers=0,
                                      drop_last=(not self.args.no_drop_last))
        dataloader_val = DataLoader(val_set, batch_size=self.args.batch_size,
                                    shuffle=True, num_workers=0,
                                    pin_memory=True,
                                    drop_last=(not self.args.no_drop_last))

        if self.args.variable_unet is True:
            network = VariableUnet(batch_norm=self.args.do_batch_norm,
                                   n_blocks=self.args.n_blocks,
                                   n_initial_block_channels=self.args.n_initial_block_channels)
        else:
            network = Unet(batch_norm=self.args.do_batch_norm,
                           p_dropout=self.args.p_dropout)

        Model = OurModel(name="unet", network=network,
                         criterion=self.args.criterion.value,
                         path_dir=self.path, lr=self.args.learning_rate,
                         batch_size=self.args.batch_size, verbose=True,
                         segmentation=True, data_trafo=dataset_aug.trafo,
                         adam_regul_factor=self.args.adam_regul_factor,
                         use_lr_scheduler=self.args.use_lr_scheduler,
                         lr_sch_patience=self.args.lr_sch_patience)

        # save a json file which indicates what parameters are used for training
        Model.save_configuration()
        Model.train(self.args.epochs, dataloader_train, validate=True,
                    dataloader_val=dataloader_val, save_observables=True)


class UnetTrainingCLI(TrainingCLI):
    training_class: type = UnetCLITraining

    def __init__(self):
        parser = ArgumentParser(
            description="Train Unet, save weights + results.")
        parser.add_argument("--p-dropout", type=float, default=0)
        parser.add_argument("--variable-unet", action='store_true')
        parser.add_argument("--n-blocks", type=int, default=4)
        parser.add_argument("--n-initial-block-channels", type=int, default=64)
        parser.add_argument("criterion", type=Criterion.__getitem__,
                            choices=Criterion)
        parser.add_argument("augmentation", type=Augmentation.__getitem__,
                            choices=Augmentation)
        super().__init__(parser)

    def default_args(self):
        args = self.get_args(
            list(Criterion)[0].name, list(Augmentation)[0].name)
        del args.criterion
        del args.augmentation
        return args

    def get_abbrev(self, args: Namespace):
        abbrev = super().get_abbrev(args)
        default = self.default_args()
        if args.p_dropout != default.p_dropout:
            abbrev += f"_do{args.p_dropout}"
        if args.variable_unet is True:
            abbrev += "_varUnet"
            if args.n_blocks != default.n_blocks:
                abbrev += f"_blk{args.n_blocks}"
            if args.n_initial_block_channels != \
                    default.n_initial_block_channels:
                abbrev += f"_ch{args.n_initial_block_channels}"
        return abbrev


def main(*args):
    cli = UnetTrainingCLI()
    training = cli.get_training()
    training.main(*args)


if __name__ == '__main__':
    main()
