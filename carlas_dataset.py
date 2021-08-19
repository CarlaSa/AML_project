import pydicom
import pylibjpeg
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, DataLoader
import os
import torch, torchvision

class my_dataset(Dataset):
    """ our dataset"""
    def __init__(self, list_files, img_size =(1000,1000)):
        self.list_files = list_files

        # transformationen 
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            #torchvision.transforms.RandomCrop(crop_size) if not center else torchvision.transforms.CenterCrop(crop_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ])

        
    def __len__(self):
        return len(list_files)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = list_files[idx]
        
        ## get file from link
        dcm_file = pydicom.read_file(img_name)
        img = dcm_file.pixel_array
        
        #### COnvert pixel_array to PIL image ####
        ### source https://stackoverflow.com/questions/42650233/how-to-access-rgb-pixel-arrays-from-dicom-files-using-pydicom

        img_2d = img.astype(float)
        img_2d_scaled = (np.maximum(img_2d,0) / img_2d.max()) * 255.0
        img_2d_scaled = np.uint8(img_2d_scaled)
        img = Image.fromarray(img_2d_scaled)
        
        ##### end ###
        
        img = self.transform(img)
        
        sample = {
            'image': img,
            }
        return sample


if __name__ == '__main__':
    list_files = []
    suffix = "dcm"
    for dirname, _, filenames in os.walk('siim-covid19-detection'):
        for filename in filenames:
            x = os.path.join(dirname, filename)
            if x.endswith(suffix):
                list_files.append(x)


    dataset = my_dataset(list_files)
    dataloader = DataLoader(dataset, batch_size=4,
                        shuffle=True, num_workers=0, )






