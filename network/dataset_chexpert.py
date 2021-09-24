import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image as Image
import torchvision.transforms as transforms

class ChestXrayDataset(Dataset):
    
    def __init__(self, folder_dir, dataframe, \
                 labels, image_size = 256):
        """
        Init Dataset
        
        Parameters
        ----------
        folder_dir: str
            folder contains all images
        dataframe: pandas.DataFrame
            dataframe contains all information of images
        image_size: int
            image size to rescale
        normalization: bool
            whether applying normalization with mean and std from ImageNet or not
        """
        #self.image_paths = [] # List of image paths
        #self.image_labels = [] # List of image labels
        
        self.folder_dir = folder_dir
        self.labels = labels
        # only use frontal images
        dataframe = dataframe[dataframe['Frontal/Lateral'] == "Frontal"]
        # only keep labels, replace nan and -1 with 0
        self.dataframe = dataframe[labels].replace(np.nan, 0).replace(-1, 0)
        
        # Define list of image transformations
        image_transformation = [
            transforms.Resize((image_size+ 20, image_size + 20)),
            transforms.RandomRotation(7),
            transforms.CenterCrop((image_size +15, image_size + 15)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.ToTensor()
        ]
        
        self.image_transformation = transforms.Compose(image_transformation)
        
    def get_loss_weights(self):
        weights = list(self.dataframe.drop(columns = ["Path"])\
                       .apply(pd.Series.sum))
        weights = [(len(self.dataframe)- w)/w for w in weights] 
        return weights
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        """
        Read image at index and convert to torch Tensor
        """
        
        # Read image
        image_path = self.folder_dir + self.dataframe.iloc[index][0]
        image_data = Image.open(image_path)#.convert("RGB") # Convert image to RGB channels
        
        # Resize and convert image to torch tensor 
        image_data = self.image_transformation(image_data)
        
        label = torch.FloatTensor(self.dataframe.iloc[index][1:])
        
        return image_data, label