
"""
This file provides some basic structures to train any model we used,
save and load the weights and test the results with the validation dataset.

"""

import torch
import torch.nn as nn
import numpy as np
import json
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix
from .losses import dice_score
from warnings import warn


def get_balanced_crossentropy_loss(dataset, verbose = False):
    """
    Returns a Crossentropy Loss with weights according to the balancing
    in the dataset

    A higher weight corresponds with a greater emphasis on this class.
    So smaller classes should get a higher weight.
    """

    if verbose:
        print("calculate balancing vector:")
    b = torch.tensor([0,0,0,0])
    for x,y in (tqdm(dataset) if verbose else dataset):
        b += y
    c_weights = (torch.sum(b)/ (4*b))
    if verbose:
        print("balancing vector will be " + str(c_weights))
    loss = nn.CrossEntropyLoss(weight= c_weights.float())
    return loss


class OurModel:
    def __init__(self,
            name,
            network,
            criterion,
            lr = 0.01,
            batch_size = None,
            path_dir = None,
            path_weights = None,
            use_cuda = torch.cuda.is_available(),
            verbose = False,
            segmentation = False,
            pretrain = False,
            data_trafo = None
            ):

        self.name = name
        self.network = network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr)
        self.criterion = criterion
        self.verbose = verbose
        self.use_cuda = use_cuda
        self.path = path_dir
        self.segmentation = segmentation
        self.lr = lr
        self.batch_size = batch_size
        self.pretrain = pretrain
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
                  "data_trafos": self.data_trafo._json_serializable()
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

    def train_one_epoch(self, dataloader):
        sum_loss = 0
        sum_dce = 0
        for x,y in (tqdm(dataloader)
                    if self.verbose and self.segmentation else dataloader):

            self.optimizer.zero_grad()

            if self.segmentation:
                y = y.float()
                y = y.squeeze()
            elif self.pretrain:
                y = y.float()
            else:
                y = torch.argmax(y.float(), dim = 1)

            x = x.float()

            if self.use_cuda:
                y = y.cuda()
                x = x.cuda()

            output = self.network(x)
            loss = self.criterion(output, y)
            loss.backward()
            sum_loss += float(torch.mean(loss)) # float() important to reduce memory!
            self.optimizer.step()

            if self.segmentation:
                dce = dice_score(torch.round(output), y, reduction='none')
                sum_dce += float(torch.mean(dce))
                #del dce
            #del output, x, y # maybe to much
        if self.segmentation:
            return sum_loss/len(dataloader), sum_dce/len(dataloader)
        else:
            return sum_loss/len(dataloader)

    def train(self, num_epochs, dataloader, validate = False,
              dataloader_val=None, save_freq = 10, save_observables = False):
        if save_observables and self.segmentation:
            losses = []
            dce_scores = []
            if validate:
                losses_val = []
                dce_scores_val =[]

        for e in (tqdm(range(1, num_epochs+1))
                        if self.verbose else range(1, num_epochs+1)):

            if self.segmentation:
                loss, dce = self.train_one_epoch(dataloader)
                if self.verbose:
                    print(f"epoch{e}: training_loss = {loss}")
                    print(f"epoch{e}: training_dice = {dce}")

                if validate:
                    loss_val, dce_val = self.val(dataloader_val)
                    if self.verbose:
                        print(f"epoch{e}: validation_loss = {loss_val}")
                        print(f"epoch{e}: validation_dice = {dce_val}")

                    # Make network trainable again
                    self.network.train()

                if save_observables:
                    losses.append(loss)
                    dce_scores.append(dce)
                    if e % save_freq == 0 or e == num_epochs:
                        np.save(f'{self.path}/loss.npy', np.array(losses))
                        np.save(f'{self.path}/dce.npy', np.array(dce_scores))
                    if validate:
                        losses_val.append(loss_val)
                        dce_scores_val.append(dce_val)
                        if e % save_freq == 0 or e == num_epochs:
                            np.save(f'{self.path}/loss_val.npy', np.array(losses_val))
                            np.save(f'{self.path}/dce_val.npy', np.array(dce_scores_val))

            else:
                loss = self.train_one_epoch(dataloader)
                if self.verbose:
                    print(f"epoch{e}: loss = {loss}")


            if e % save_freq == 0:
                self.save_weights(e)


    def val(self, dataloader_val):
        if not self.segmentation:
            self.network.eval()
            acc = 0
            y_true = []
            y_pred = []

            with torch.no_grad():
                for x,y in tqdm(dataloader_val):
                    y = torch.argmax(y.float(), dim = 1)
                    x = x.float()

                    if self.use_cuda:
                        y = y.cuda()
                        x = x.cuda()

                    output = self.network(x)
                    output = torch.argmax(output, dim = 1)
                    y_true.extend(y.tolist())
                    y_pred.extend(output.tolist())
                    acc += sum(output == y)
                #print("overall correctly classified: " + str(acc / len(val_set)))
                print(confusion_matrix(y_true, y_pred))

        else:
            self.network.eval()
            sum_loss = 0
            sum_dce = 0

            with torch.no_grad():
                for x,y in tqdm(dataloader_val):
                    y = y.float()
                    y = y.squeeze()
                    x = x.float()

                    if self.use_cuda:
                        y = y.cuda()
                        x = x.cuda()

                    output = self.network(x)
                    loss = self.criterion(output, y)
                    sum_loss += float(loss)
                    dce = dice_score(torch.round(output), y, reduction='none')
                    sum_dce += float(torch.mean(dce))
                    del x,y, loss, dce, output # maybe too much
            return sum_loss/len(dataloader_val), sum_dce/len(dataloader_val)
