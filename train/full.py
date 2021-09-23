import os
import re
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import torch
from torch import nn
from argparse import ArgumentParser

from datasets import LoadDataset, CustomOutput, Knit
from datasets.custom_output import image_tensor, study_label_5

from network.unet import Unet
from network.feature_extractor import ResNet, ResnetOriginal
from network.full_model import EndNetwork, FullModel
from network.training import FullTraining, get_balanced_crossentropy_loss

from .common_train import TrainingCLI, Augmentation, CLITraining
from utils.device import device


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)


class FullCLITraining(CLITraining):
    def run(self):
        super().run()
        print("debug augmentation:", self.args.augmentation)

        # Get Data
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

        # Get Networks
        unet = Unet(batch_norm=self.args.do_batch_norm)
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

        if resnet_config["network"] == "ResnetOriginal":
            rn = [int(n) for n
                  in re.search(r"resnet(\d+)", resnet_config_file).group(1)]
            resnet = ResNet(rn, out_shape=self.args.resnet_out_shape)
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
            resnet_type = resnet_config["type"]
            shapes = list(resnet_config["shapes"])
            trainable_level = int(resnet_config["trainable_level"])
            trainable_resnet = bool(resnet_config["trainable_resnet"])
            resnet = ResnetOriginal(type=resnet_type, shapes=shapes,
                                    trainable_resnet=trainable_resnet,
                                    trainable_level=trainable_level)
            resnet.load_state_dict(torch.load(self.args.resnet_weights,
                                              map_location=device))
            resnet.fc.block = nn.Sequential(*list(resnet.fc.block.children())
                                            [:-self.args.resnet_fc_cutoff])

        # Full network
        end_network = EndNetwork(features_shape=self.args.feature_shape)

        model = FullModel(unet=unet, feature_extractor=resnet, end=end_network,
                          threshold=0.5, unet_trainable=False,
                          feature_extractor_trainable=False)

        print("Number of trainable parameters:",
              sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("Number of parameters:",
              sum(p.numel() for p in model.parameters()))

        loss = get_balanced_crossentropy_loss(train_set, verbose=True, shape=5)
        training = FullTraining(self.path + "/", model, loss,
                                batch_size=self.args.batch_size,
                                verbose_level=2, path_dir=".", data_trafo=None)
        training.train(self.args.epochs, dataloader=dataloader_train,
                       dataloader_val=dataloader_val, validate=True,
                       save_observables=True, det_obs_freq=0)


class FullTrainingCLI(TrainingCLI):
    training_class: type = FullCLITraining

    def __init__(self):
        parser = ArgumentParser(
            description="Train full network, save weights + results.")
        parser.add_argument("--unet-weights", type=file_path)
        parser.add_argument("--resnet-weights", type=file_path)
        parser.add_argument("--augmentation", type=Augmentation.__getitem__,
                            choices=Augmentation, default=Augmentation.NA)
        parser.add_argument("--feature-shape", type=int, default=512)
        parser.add_argument("--resnet-out-shape", type=int)
        parser.add_argument("--resnet-fc-cutoff", type=int)
        parser.add_argument("--path-prefix", default="_full_training")
        super().__init__(parser)


def main(*args):
    cli = FullTrainingCLI()
    training = cli.get_training()
    training.main(*args)


if __name__ == '__main__':
    main()
