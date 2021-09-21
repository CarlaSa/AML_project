import torch
import torch.nn as nn
import numpy as np
import json
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix
from .losses import dice_score
from warnings import warn
import random
import re

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def get_balanced_crossentropy_loss(dataset, verbose = False, shape = 4):
    """
    Returns a Crossentropy Loss with weights according to the balancing
    in the dataset

    A higher weight corresponds with a greater emphasis on this class.
    So smaller classes should get a higher weight.
    """

    if verbose:
        print("calculate balancing vector:")
    b = torch.tensor([0]* shape)
    for x,y in (tqdm(dataset) if verbose else dataset):
        b += y
    c_weights = (torch.sum(b)/ (4*b))
    if verbose:
        print("balancing vector will be " + str(c_weights))
    loss = nn.CrossEntropyLoss(weight= c_weights.float())
    return loss

class BaseTraining:
    def __init__(self,
            name,
            network,
            criterion,
            lr = 0.01,
            batch_size = None,
            path_dir = None,
            path_weights = None,
            use_cuda = torch.cuda.is_available(),
            verbose_level = 0,
            data_trafo = None,
            adam_regul_factor = 0.
            ):

        # define model
        self.name = name
        self.network = network
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr)
        self.adam_regul_factor = adam_regul_factor
        self.criterion = criterion
        self.batch_size = batch_size


        self.verbose = verbose_level
        self.use_cuda = use_cuda
        self.path = path_dir
        self.data_trafo = data_trafo
        self.epochs = 0
        self.start_epoch = 0

        self.observables = {"loss": []
                            }

        if batch_size is None:
            warn("Batch size is not specified."
                  + " It can't be stored in net_config.json")
        if data_trafo is None:
            warn("Data trafos are not specified."
                  + " It can't be stored in net_config.json")

        if self.use_cuda:
            self.network = self.network.cuda()
            self.criterion = self.criterion.cuda()

        if path_weights is not None:
            load_weights(path_weights)

    def save_configuration(self):
        config = {"network": self.network.__class__.__name__,
                  "optimizer": self.optimizer.__class__.__name__,
                  "adam_regul_factor": self.adam_regul_factor,
                  "batch_size": self.batch_size,
                  "learning_rate": self.lr,
                  "loss": self.criterion.__class__.__name__,
                  #"data_trafos": self.data_trafo._json_serializable(),
                  **getattr(self.network, "hyperparameters", {})
                  }
        if self.data_trafo is not None:
            config["data_trafo"] = self.data_trafo._json_serializable()

        with open(f'{self.path}/{self.name}_net_config.json', 'w') as file:
            json.dump(config, file)

    def load_weights(self, path_end):
        if self.path is not None:
            path = self.path + path_end
        else:
            path = path_end
        self.network.load_state_dict(torch.load(path))
        try:
            pattern = re.compile(r"^.*_e(\d+).*$")
            self.start_epoch = int(pattern.match(path_end).group(1))
        except:
            warn("Could not extract epoch from weights.")

    def save_weights(self):
        if self.path is not None:
            path = f'{self.path}/{self.name}_e{self.epochs}.ckpt'
        else:
            path = f'./{self.name}_e{self.epochs}.ckpt'
        torch.save(self.network.state_dict(), path)

    def _preprocess(self,x,y):
        """
        Things that need to be done to x and y before training
        """
        raise NotImplementedError

    def _evaluation_methods(self, y_true, y_pred):
        # if opened first time
        if not "acc" in dir(self):
            self.acc = 0

        self.acc += sum(y_pred == y_true)

    def print_observables(self):
        """
        print the observables (e.g. losses)
        """
        for item in self.observables.items():
            try:
                print(f"epoch{self.epochs}: {item[0]} = {item[1][-1]}")
            except:
                print(f"epoch{self.epochs}: {item[0]} = no value stored")

    def train_one_epoch(self, dataloader):
        sum_loss = 0
        for x,y in (tqdm(dataloader)
                    if self.verbose == 2 else dataloader):

            self.optimizer.zero_grad()

            x,y = self._preprocess(x,y)

            if self.use_cuda:
                y = y.cuda()
                x = x.cuda()

            output = self.network(x)
            loss = self.criterion(output, y)
            loss.backward()

            sum_loss += float(torch.mean(loss))

            self.optimizer.step()
        return sum_loss/len(dataloader)

    def train(self, num_epochs, dataloader, validate = False,
              dataloader_val=None, save_freq = 10, save_observables = False):

        if validate:
            self.observables["loss_val"] = []

        self.network.train()
        start = self.start_epoch + 1
        end = self.start_epoch + num_epochs + 1
        for e in (tqdm(range(start, end)) if self.verbose > 0
                  else range(start, end)):

            self.epochs = e

            loss = self.train_one_epoch(dataloader)
            self.observables["loss"].append(loss)
            if validate:
                self.validate(dataloader_val)
            if e % save_freq == 0:
                self.save_weights()
                if save_observables:
                    for item in self.observables.items():
                        np.save(f"{self.path}/{self.name}_{item[0]}.npy", np.array(item[1]))
            if self.verbose > 0:
                self.print_observables()


    def validate(self, dataloader_val):
        self.network.eval()

        with torch.no_grad():
            sum_val_loss = 0
            for x,y in tqdm(dataloader_val):

                x,y = self._preprocess(x,y)
                if self.use_cuda:
                    y = y.cuda()
                    x = x.cuda()

                output = self.network(x)
                loss_val = self.criterion(output, y)
                sum_val_loss += float(torch.mean(loss_val))
            self.observables["loss_val"].append(sum_val_loss/len(dataloader_val))
                #self._evaluation_methods(output, y)


class PretrainTraining(BaseTraining):
    def _preprocess(self, x, y):
        y = y.float()
        x = x.float()
        return x,y

    def _evaluation_methods(self, y_true, y_pred):
        # if opened first time
        if not "acc" in dir(self):
            self.acc = 0

        self.acc += torch.mean((y_true-y_pred)**2)


class UnetTraining(BaseTraining):
    def _preprocess(self, x, y):
        y = y.float()
        y = y.squeeze()
        x = x.float()
        return x, y

class FullTraining(BaseTraining):
    def _preprocess(self, x, y):
        y = y.float()
        x = x.float()
        return x,y
