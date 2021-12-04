import torch
import torch.nn as nn
from torchvision.models import resnet18

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os

PRETRAINED = resnet18(pretrained=True)


class ImageModel(nn.Module):
    def __init__(self, pretrained=PRETRAINED):
        super(ResNet18, self).__init__()
        self.pretrained = pretrained
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(1000, 128)

    def forward(self, x):
        out = self.pretrained(x)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class CompareNet(nn.Module):
    def __init__(self):
        super(CompareNet, self).__init__()
        self.before_net = ImageModel()
        self.after_net = ImageModel()
        self.cosine = nn.CosineSimilarity(dim=1)

    def forward(self, before_input, after_input):
        before = self.before_net(before_input) 
        after = self.after_net(after_input)
        out = self.cosine(before, after)
        return out


# class RNNwithImageModel(nn.Module):
#     def __init__(
#         self,
#         device,
#         image_model,
#         input_size,
#         hidden_size,
#         seq_length,
#         num_layers=1,
#         num_classes=1
#         ):
#         super(RNNwithImageModel, self).__init__()
#         self.device = device
#         self.image_model = image_model
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.seq_length = seq_length
#         self.num_layers = num_layers
#         self.num_classes = num_classes
#         self.lstm = nn.LSTM(
#             input_size=self.input_size,
#             hidden_size=self.hidden_size,
#             num_layers=self.num_layers,
#             batch_first=True
#         )


#     def forward(self, before_input, after_input):
#         before = self.image_model(before_input)
#         after = self.image_model(after_input)
#         h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
#         c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

