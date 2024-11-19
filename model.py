import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights

class MammographyModel(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        
        self.rnet = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.rnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.rnet.fc = nn.Identity()       
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)
        
    def forward(self, img):
        resnet_out = self.rnet(img)
        
        x_final = resnet_out.to(torch.float32)
        x_final = self.fc1(x_final)
        x_final = self.dropout(x_final)
        x_final = torch.relu(x_final)
        
        x_final = self.fc2(x_final)
        x_final = self.dropout(x_final)
        x_final = torch.relu(x_final)  
        x_final = self.fc3(x_final)
        
        return x_final