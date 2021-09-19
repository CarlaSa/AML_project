import torch
import torch.nn as nn
import numpy as np
import json
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix
from .losses import dice_score
from warnings import warn


np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

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
            data_trafo = None
            ):

        # define model
        self.name = name
        self.network = network
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr)
        self.criterion = criterion
        self.batch_size = batch_size


        self.verbose = verbose_level
        self.use_cuda = use_cuda
        self.path = path_dir
        self.data_trafo = data_trafo

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
                  "batch_size": self.batch_size,
                  "learning_rate": self.lr,
                  "loss": self.criterion.__class__.__name__,
                  "data_trafos": self.data_trafo._json_serializable(),
                  **getattr(self.network, "hyperparameters", {})
                  }
        with open(f'./{self.path}/net_config.json', 'w') as file:
            json.dump(config, file)

    def load_weights(self, path_end):
        if self.path is not None:
            path = self.path + path_end
        else:
            path = path_end
        self.network.load_state_dict(torch.load(path))

    def save_weights(self, epoch):
        if self.path is not None:
            path = f'{self.path}/{self.name}_e{epoch}.ckpt'
        else:
            path = f'./{self.name}_e{epoch}.ckpt'
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

    def _print_eval(self):
        print(self.acc)

        # reset variable
        del self.acc

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

    def train(self, num_epochs, dataloader, save_freq = 10):

        losses = []

        self.network.train()
        for e in (tqdm(range(1, num_epochs+1)) if self.verbose > 0 
                else range(1, num_epochs+1)):

            loss = self.train_one_epoch(dataloader)
            if self.verbose > 0:
                print(f"epoch{e}: loss = {loss}")
            if e % save_freq == 0:
                self.save_weights(e)

            losses.append(loss)
        return losses


    def validate(self, dataloader_val):
        self.network.eval()

        with torch.no_grad():
            for x,y in tqdm(dataloader_val):

                x,y = self._preprocess(x,y)
                if self.use_cuda:
                    y = y.cuda()
                    x = x.cuda()

                output = self.network(x)
                self._evaluation_methods(output, y)

        self._print_eval()


class PretrainTraining(BaseTraining):
    def _preprocess(self, x, y):
        y = torch.argmax(y.float(), dim = 1)
        x = x.float()
        return x,y


class UnetTraining(BaseTraining):
    def _preprocess(self, x, y):
        y = y.float()
        y = y.squeeze()
        x = x.float()
        return x, y

