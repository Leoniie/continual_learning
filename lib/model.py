import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
matplotlib.style.use('ggplot')

class MLP(torch.nn.Module):
    '''
    Mulitlayer Perceptron, all linear
    Acitvations: ReLu

    Args:
        dimensions: List of integers specifying the layer dimensions. Example: 724, 200, 10 would mean layer 1(724,200), layer2(200,10)
    '''


    def __init__(self, dimensions, nonlinearity):
        super().__init__()

        
        self.dimensions = dimensions
        self.nonlinearity = nonlinearity

        self.layers = torch.nn.ModuleList()
        for k in range(len(dimensions)-1):
            self.layers.append(torch.nn.Linear(dimensions[k], dimensions[k+1]))
    
    def forward(self, x):
        """
        Defines the computation performed at every call.
        """
        x = x.view(-1, self.dimensions[0]) # Flatten the input

        for l in self.layers[:-1]:
            x = self.nonlinearity(l(x))
        
        y_pred = self.layers[-1](x)
        return y_pred




class VIBModel(torch.nn.Module):

    def __init__(self, encoder, decoder_class, decoder_rec):
        super().__init__()
        self.encoder = encoder
        self.decoder_class = decoder_class
        self.decoder_rec = decoder_rec

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self,x):
        mu, logvar = encoder(x)
        z = self.reparameterize(mu, logvar)
        pred = torch.softmax(self.decoder_class(z))
        recon = torch.sigmoid(self.decoder_rec(z))
        return pred, recon, mu, logvar

