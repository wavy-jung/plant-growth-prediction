import torch
import torch.nn as nn
from torchvision.models import resnet18

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os

class ImageModel(nn.Module):
    def __init__(self, pretrained):
        super(ResNet18, self).__init__()
        self.pretrained = pretrained
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(1000, 128)

    def forward(self, x):
        out = self.pretrained(x)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class RNNwithImageModel(nn.Module):
    def __init__(
        self,
        image_model : ImageModel,
        input_size : int,
        hidden_size : int,
        
        ):
        super(RNNwithImageModel, self).__init__()
