
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix


def get_balanced_crossentropy_loss(dataset, verbose = False):
    """
    returns a Crossentropy Loss with weights according to the balancing in the dataset
    
    A higher weight corresponds with a greater emphasis on this class. So smaller classes 
    should get a higher weight.
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
            path = None,
            use_cuda = torch.cuda.is_available(),
            verbose = False
            ):
        
        self.name = name
        self.network = network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr)
        self.criterion = criterion
        self.verbose = verbose
        self.use_cuda = use_cuda
        self.path = path

        if self.use_cuda:
            self.network = self.network.cuda()
        
        if path is not None:
            load_weights(path)
        
    def load_weights(self, path_end):
        if self.path not None:
            path = self.path + path_end
        else:
            path = path_end
        self.network.load_state_dict(torch.load(path))
        
    def save_weights(self, epoch):
        if self.path not None:
            path = f'{self.path}/{self.name}_e{epoch}.ckpt'
        else:
            path = f'./{self.name}_e{epoch}.ckpt'
        torch.save(network.state_dict(), path)
        
    def train_one_epoch(self, dataloader):
        sum_loss = 0
        for x,y in tqdm(dataloader):
            self.optimizer.zero_grad()
            y = torch.argmax(y.float(), dim = 1)
            x = x.float()

            if self.use_cuda:
                y = y.cuda()
                x = x.cuda()
        
            output = self.network(x)
        
            loss = self.criterion(output, y)
            loss.backward()
            sum_loss += torch.mean(loss)
            self.optimizer.step()
        return sum_loss/len(dataloader)
            
    def train(self, num_epochs, dataloader, save_freq = 5):
        for e in (tqdm(range(num_epochs)) if self.verbose else range(num_epochs)):
            loss = self.train_one_epoch(dataloader)
            if self.verbose:
                print(f"epoch{e}: loss = {loss}")
            if (e-1) % save_freq == 0:
                self.save_weights(e)
        
    def val(self, dataloader_val):
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