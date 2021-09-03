import torch.nn as nn
import torch.nn.functional as F
import torch

class vanilla(nn.Module):
    def __init__(self, input_size = 256, ):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3) 
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(32, 32, 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(32, 64, 3)
        
        after_conv_size = int((input_size / 16 - 4)**2  * 64)
        self.fc1 = nn.Linear(after_conv_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        # B:= Batchsize
        # Inputsize: HxW
        # shape x: (B, 1, H, W)
        
        x = self.conv1(x)  # shape: (B, 16, H - 2 , W - 2 )
        x = self.pool1(x)  # shape: (B, 16, H/2 - 1 , W/2 - 1 )
        
        x = F.relu(x)
        x = self.conv2(x)  # shape: (B, 16, H/2 - 3, W/2 - 3)
        x = self.pool2(x)  # shape: (B, 16, H/4 - 2, W/4 - 2)
        
        x = F.relu(x)
        x = self.conv3(x)  # shape: (B, 32, H/4 - 4 , W/4 - 4)
        x = self.pool3(x)  # shape: (B, 32, H/8 - 2 , W/8 - 2)
        
        x = F.relu(x)
        x = self.conv4(x)  # shape: (B, 32, H/8 - 4 , W/8 - 4)   
        x = self.pool4(x)  # shape: (B, 32, H/16 - 2 , W/16 - 2)
        
        x = F.relu(x)
        x = self.conv5(x)  # shape: (B, 64, H/16 - 4 , W/16 - 4)
        
        x = torch.flatten(x, 1) # shape: (B, rest)
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x