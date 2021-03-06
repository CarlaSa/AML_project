import os
import re
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import torch
from torch import nn
from argparse import ArgumentParser, Namespace

from datasets import LoadDataset, CustomOutput, Knit
from datasets.custom_output import image_tensor, study_label_5

from network.unet import Unet
from network.variable_unet import Unet as VUnet
from network.feature_extractor import ResNet, ResnetOriginal
from network.full_model import EndNetwork, FullModel, EndNetwork_minimal
from network.training import FullTraining

from .common_train import TrainingCLI, Augmentation, CLITraining, ArgparseEnum
from utils.device import device


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)


CEBAL_WEIGHTS = torch.tensor([0.9078, 0.5553, 1.5064, 4.0752, 5.3061])


class Criterion(ArgparseEnum):
    BCE = nn.BCELoss().cuda()
    CE = nn.CrossEntropyLoss().cuda()
    CEBAL = nn.CrossEntropyLoss(weight=CEBAL_WEIGHTS).cuda()


class FullCLITraining(CLITraining):
    def run(self):
        super().run()
        model = self.get_model()
        dataloader_train, dataloader_val = self.get_dataloaders()

        print("Number of trainable parameters:",
              sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("Number of parameters:",
              sum(p.numel() for p in model.parameters()))

        training = FullTraining(self.path + "/", model,
                                criterion=self.args.criterion.value,
                                batch_size=self.args.batch_size,
                                verbose_level=2, path_dir=".", data_trafo=None,
                                use_lr_scheduler=self.args.use_lr_scheduler,
                                lr_sch_patience=self.args.lr_sch_patience,
                                lr=self.args.learning_rate,
                                adam_regul_factor=self.args.adam_regul_factor,
                                early_stopping=self.args.early_stopping)
        training.train(self.args.epochs, dataloader=dataloader_train,
                       dataloader_val=dataloader_val, validate=True,
                       save_observables=True, det_obs_freq=0)

    def get_dataloaders(self):
        loaded_data = LoadDataset("_data/preprocessed256_new",
                                  image_dtype=float, label_dtype=float)
        knit_data = Knit(loaded_data, study_csv="_data/train_study_level.csv",
                         image_csv="_data/train_image_level.csv")
        dataset_plain = CustomOutput(knit_data, image_tensor, study_label_5)
        dataset_aug = CustomOutput(knit_data, image_tensor, study_label_5,
                                   trafo=self.args.augmentation.value)
        dataset_aug.trafo.max_transformands = 1

        l_data = len(dataset_aug)
        indices = list(range(l_data))
        train_size = self.args.absolute_training_size
        val_size = l_data - train_size
        print("Training: ", train_size, "Validation: ", val_size)

        train_indices, val_indices = train_test_split(indices, random_state=4,
                                                      train_size=train_size,
                                                      test_size=val_size)

        train_set = Subset(dataset_aug, train_indices)
        val_set = Subset(dataset_plain, val_indices)

        dataloader_train = DataLoader(train_set,
                                      batch_size=self.args.batch_size,
                                      shuffle=True, num_workers=0)
        dataloader_val = DataLoader(val_set,
                                    batch_size=self.args.batch_size,
                                    shuffle=True, num_workers=0,
                                    pin_memory=True)
        return dataloader_train, dataloader_val

    def get_model(self):
        # Get Networks
        unet_dir = os.path.dirname(self.args.unet_weights)
        with open(os.path.join(unet_dir, "net_config.json")) as config_file:
            unet_config = json.load(config_file)
        if "n_blocks" in unet_config or "n_initial_block_channels" in unet_config:
            n_blk = unet_config["n_blocks"]
            n_in_ch = unet_config["n_initial_block_channels"]
            unet = VUnet(batch_norm=unet_config["batch_norm"],
                         n_blocks=n_blk, n_initial_block_channels=n_in_ch)
        else:
            unet = Unet(batch_norm=unet_config["batch_norm"])
        unet.load_state_dict(torch.load(self.args.unet_weights,
                                        map_location=device))

        # Get ResNet
        pattern = re.compile(r"(_e(\d+)\.ckpt)$")
        load_epoch = int(pattern.search(self.args.resnet_weights).group(2))
        print("Attempting to load resnet from epoch", load_epoch)
        resnet_config_file = pattern.sub("_net_config.json",
                                         self.args.resnet_weights)

        with open(resnet_config_file) as f:
            resnet_config = json.load(f)

        sigmoid_activation = self.args.resnet_no_sigmoid_activation is not True
        if resnet_config["network"] == "ResNet":
            dims = [int(n) for n
                    in re.search(r"resnet(\d+)", resnet_config_file).group(1)]
            print("trying to load", f"dims={dims}",
                  f"out_shape={self.args.resnet_out_shape}")
            resnet = ResNet(dims, out_shape=self.args.resnet_out_shape,
                            sigmoid_activation=sigmoid_activation)
            resnet.load_state_dict(torch.load(self.args.resnet_weights,
                                              map_location=device))
            children = list(resnet.end.children())
            # cut off everything from last Linear layer:
            for i, child in enumerate(reversed(children)):
                if isinstance(child, nn.Linear):
                    resnet.end = nn.Sequential(*children[:-1-i])
                    break
            else:
                raise RuntimeError("No nn.Linear found in ResNet")
        elif resnet_config["network"] == "ResnetOriginal":
            if self.args.resnet_fc_cutoff is None:
                raise RuntimeError("Please specify --resnet-fc-cutoff=n")
            resnet_type = resnet_config["type"]
            shapes = list(resnet_config["shapes"])
            trainable_level = int(resnet_config["trainable_level"])
            trainable_resnet = bool(resnet_config["trainable_resnet"])
            resnet = ResnetOriginal(type=resnet_type, shapes=shapes,
                                    trainable_resnet=trainable_resnet,
                                    trainable_level=trainable_level,
                                    sigmoid_activation=sigmoid_activation)
            resnet.load_state_dict(torch.load(self.args.resnet_weights,
                                              map_location=device))
            resnet.fc.block = nn.Sequential(*list(resnet.fc.block.children())
                                            [:-self.args.resnet_fc_cutoff])

        # Full network
        use_dropout = self.args.no_dropout is not True
        if self.args.endnet_minimal is True:
            end_network = EndNetwork_minimal(
                features_shape=self.args.feature_shape,
                latent_shape=self.args.latent_shape, use_dropout=use_dropout,
                use_dropout_conv=self.args.dropout_conv is True,
                use_batchnorm=self.args.do_batch_norm is True)
        else:
            end_network = EndNetwork(features_shape=self.args.feature_shape,
                                     use_dropout=use_dropout)

        return FullModel(unet=unet, feature_extractor=resnet, end=end_network,
                         threshold=0.5,
                         unet_trainable=self.args.unet_trainable,
                         feature_extractor_trainable=self.args.resnet_trainable)


class FullTrainingCLI(TrainingCLI):
    training_class: type = FullCLITraining

    def __init__(self):
        parser = ArgumentParser(
            description="Train full network, save weights + results.")
        parser.add_argument("--unet-weights", type=file_path)
        parser.add_argument("--resnet-weights", type=file_path)
        parser.add_argument("--augmentation", type=Augmentation.__getitem__,
                            choices=Augmentation, default=Augmentation.NA)
        parser.add_argument("--criterion", type=Criterion.__getitem__,
                            choices=Criterion, default=Criterion.CEBAL)
        parser.add_argument("--feature-shape", type=int, default=512)
        parser.add_argument("--resnet-out-shape", type=int)
        parser.add_argument("--resnet-fc-cutoff", type=int)
        parser.add_argument("--resnet-no-sigmoid-activation",
                            action="store_true")
        parser.add_argument("--no-dropout", action="store_true")
        parser.add_argument("--resnet-trainable", action="store_true")
        parser.add_argument("--unet-trainable", action="store_true")
        parser.add_argument("--path-prefix", default="_full_training")
        parser.add_argument("--endnet-minimal", action="store_true")
        parser.add_argument("--latent-shape", type=int, default=64)
        parser.add_argument("--dropout-conv", action="store_true")
        parser.add_argument("--early-stopping", type=int)
        super().__init__(parser)

    def get_abbrev(self, args: Namespace):
        abbrev = super().get_abbrev(args)
        if args.early_stopping is not None:
            abbrev += f"_es{args.early_stopping}"
        if args.resnet_no_sigmoid_activation is True:
            abbrev += "_nosig"
        if args.no_dropout is True:
            abbrev += "_nodo"
        if args.resnet_fc_cutoff is not None:
            abbrev += f"_fcc{args.resnet_fc_cutoff}"
        if args.endnet_minimal is True:
            abbrev += "_mini"
            abbrev += f"_ls{args.latent_shape}"
            abbrev += "_doconv" if args.dropout_conv is True else "_nodoconv"
        return abbrev


def main(*args):
    cli = FullTrainingCLI()
    training = cli.get_training(*args)
    training.main()


if __name__ == '__main__':
    main()
