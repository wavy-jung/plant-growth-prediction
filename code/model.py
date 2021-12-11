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
        pretrained='regnetx_016'
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
        model_name='regnetx_016'
        ):
        super(CompareNet, self).__init__()
        self.before_net = ImageModel(dim_embedding=dim_embedding, pretrained=model_name)
        self.after_net = ImageModel(dim_embedding=dim_embedding, pretrained=model_name)
        self.model_name = model_name
        self.num_input = 2*dim_embedding
        self.fc = nn.Linear(self.num_input, 1)


    def forward(self, before_input, after_input):
        before = self.before_net(before_input) 
        after = self.after_net(after_input)
        out = torch.cat((before, after), dim=-1)
        out = self.fc(out)
        return out