import torch
import torch.nn as nn
from torchvision import models
import timm

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os


class ImageModel(nn.Module):

    def __init__(
        self,
        dim_embedding=8,
        pretrained='regnetx_004'
        ):
        super(ImageModel, self).__init__()
        self.pretrained = timm.create_model(pretrained, pretrained=True, num_classes=dim_embedding)
        self.dim_embedding = dim_embedding


    def forward(self, x):
        out = self.pretrained(x)
        return out



class CompareNet(nn.Module):

    def __init__(
        self,
        dim_embedding=8,
        model_name='regnetx_004'
        ):
        super(CompareNet, self).__init__()
        self.before_net = ImageModel(dim_embedding=dim_embedding, pretrained=model_name)
        self.after_net = ImageModel(dim_embedding=dim_embedding, pretrained=model_name)
        self.num_input = 2*dim_embedding
        self.fc = nn.Linear(self.num_input, 1)


    def forward(self, before_input, after_input):
        before = self.before_net(before_input) 
        after = self.after_net(after_input)
        out = torch.cat((before, after), dim=-1)
        out = self.fc(out)
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

