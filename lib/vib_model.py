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

import sys
import math

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tensorflow.python.training.input import batch
matplotlib.style.use('ggplot')

from lib import utils

class ENCODER(torch.nn.Module):
    """
    Encoder (one single Encoder used for all tasks)
    forward method returns encoding z (dimension: n_gaussian), also mu and logvar to compute loss

    """    
    def __init__(self, dimensions, nonlinearity):

        """
        Args: 
           
            dimensions: List of layer dimensions. The last layer will be added twice for the reparametrization trick. E.g. dimensions (784, 400, 400, 100) will lead to matrices 784 x 400, 400 x 400 and twice 400 x 100, 400 x 100.
            
            nolinearity: nonlinearity applied between the layers
        """
        super(ENCODER, self).__init__()
        
        self.dimensions = dimensions
        self.nonlinearity = nonlinearity

        self.layers = torch.nn.ModuleList()
        for k in range(len(dimensions)-1):
            self.layers.append(torch.nn.Linear(dimensions[k], dimensions[k+1]))
        
        # append the last layer twice for the reparameterization trick.
        self.layers.append(torch.nn.Linear(dimensions[len(dimensions)-2], dimensions[len(dimensions)-1]))
    
    def forward(self, x):
        """
        Defines the computation performed at every call.
        """
        x = x.view(-1, self.dimensions[0]) # Flatten the input

        for l in self.layers[:-2]:
            x = self.nonlinearity(l(x))
        
        mu = self.layers[-2](x)
        logvar = self.layers[-1](x)
        
        return mu, logvar





class DECODER(torch.nn.Module):

    def __init__(self, dimensions, nonlinearity):
        """

        DECODER - one decoder per task. Takes encoding z as input and generates reconstruction x' that has to match x. Training involves backpropagation of the loss w.r.t. to both the encoder and the decoder. 
        Args: 
            
            n_gaussian: number of nodes in the innermost random gaussian layer 
                (number of dimenions to generate random samples with the decoder)
            n_hidden: number of nodes in the hidden layers
            nonlinearity: activation function for forward method, default relu
        """
        super(DECODER, self).__init__()

        self.dimensions = dimensions
        self.nonlinearity = nonlinearity

        self.layers = torch.nn.ModuleList()
        for k in range(len(dimensions)-1):
            self.layers.append(torch.nn.Linear(dimensions[k], dimensions[k+1]))
    

    def forward(self, x):
        """Decodes the gaussian units back to image"""

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
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        #pred = torch.nn.functional.softmax(self.decoder_class(z), dim=1)
        pred = self.decoder_class(z)
        recon = torch.sigmoid(self.decoder_rec(z))
        return pred, recon, mu, logvar


# Vib model without reconstruction decoder
class VIBModel_class(torch.nn.Module):

    def __init__(self, encoder, decoder_class):
        super().__init__()
        self.encoder = encoder
        self.decoder_class = decoder_class

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self,x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        #pred = torch.nn.functional.softmax(self.decoder_class(z), dim=1)
        pred = self.decoder_class(z)
        recon = x
        return pred, recon, mu, logvar

# def loss_function_vib_recon(recon_x, x,  mu, logvar, pred, target, beta_1, beta_2, soft):
#     """Computer the loss in case of substraction of beta_1 (reconstruction).
    
#     Args:
#         recon_x: image reconstructed by reconstruction decoder
#         pred: label prediction made by classification decoder
#         x: original input image
#         mu, logvar: encoding after x passed through encoder, used for reparameterization
#         beta_1, beta_2: factors used for VIB objective. Different loss function is called depending on whether beta_1 is substracted (reconstruction) or added (prior matching)
    
#     Returns: (batch) loss
#     """
#     criterion = torch.nn.CrossEntropyLoss()
#     BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
#     if (soft is False): 
#         loss_class = F.cross_entropy(pred, target, reduction = 'sum')
#     else: 
#         loss_class = utils.soft_cross_entropy(pred, target, pred.shape[1], reduction = 'sum')

#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     loss_rec = BCE + KLD
#     loss_tot = loss_class - beta_1 * BCE + beta_2 * KLD

#     return loss_tot, loss_class, loss_rec


# def loss_function_vib_prior(mu, logvar, pred, target, beta_1, beta_2, soft):
#     """Compute the loss in case of addition of beta_1 (prior matching).
    
#     Args:
#         x: original input image
#         mu, logvar: encoding after x passed through encoder, used for reparameterization
#         beta_1, beta_2: factors used for VIB objective. Different loss function is called depending on whether beta_1 is substracted (reconstruction) or added (prior matching)
    
#     Returns:  (batch) loss
#     """

#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     if (soft == False): 
#         loss_class = F.cross_entropy(pred, target, reduction = 'sum')
#     else: loss_class = utils.soft_cross_entropy(pred, target, pred.shape[1], reduction = 'sum')

#     loss_tot = loss_class + (beta_1 + beta_2) * KLD
#     return loss_tot, loss_class, KLD


def loss_function_vib(recon_x, x,  mu, logvar, pred, target, beta_1, beta_2, soft = False, model_class = False):
    
    """Calculates the loss function for the VIB objective (to MINIMIZE). Different loss functions depending on whether beta_1 is substracted (reconstruction) or added (prior matching)
    Args:

        recon_x: image reconstructed by reconstruction decoder
        pred: label prediction made by classification decoder
        x: original input image
        mu, logvar: encoding after x passed through encoder, used for reparameterization
        beta_1, beta_2: factors used for VIB objective. Different loss function is called depending on whether beta_1 is substracted (reconstruction) or added (prior matching)
        model_class: is true if we are only dealing with a classifier model, i.e. it has only two parts and not the reconstruction decoder
    
    Returns: 
        Calls loss function of the joint model (encoder, reconstrcution decoder, classification decoder)
    """

    x = x.view(-1, 784)

    if (soft is False): 
            loss_class = F.cross_entropy(pred, target, reduction = 'sum')
    else: 
        loss_class = utils.soft_cross_entropy(pred, target, pred.shape[1], reduction = 'sum')
    
    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if (model_class): 
        loss_tot = loss_class + beta_2 * KLD
        #loss_tot = loss_class
        return loss_tot, loss_class, KLD # we only need loss class + KLD here

    if (beta_1 <= 0): 
        # Compute the loss in case of substraction of beta_1 (reconstruction).
        if (math.isnan(torch.max(x)) or math.isnan(torch.max(recon_x))): 
            return None, None, None
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        loss_rec = BCE + KLD
        loss_tot = loss_class - beta_1 * BCE + beta_2 * KLD
        return loss_tot, loss_class, loss_rec
    else: 
        # Compute the loss in case of addition of beta_1 (prior matching)
        loss_tot = loss_class + (beta_1 + beta_2) * KLD
        return loss_tot, loss_class, KLD

@torch.no_grad()
def accuracy(model, test_loader, device):
    """
    Compute accuracy on test set.
    """
    # Prepare the model for testing
    model.train(mode=False)
    model.eval()
    tot_test, tot_acc = 0.0, 0.0
    # Count correct predictions for all data points in test set
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        out = model(x_batch)[0]
        # Batch accuracy
        pred = torch.max(out, dim=1)[1]
        acc = pred.eq(y_batch).sum().item()
        tot_acc += acc
        tot_test += x_batch.size()[0]
    return tot_acc / tot_test


def train_model_class(model, train_loader, replay_train_loader, optimizer, device, beta_1, beta_2, replay_weight, replay):
    model.train()
    train_loss, class_loss, rec_loss = 0.0, 0.0, 0.0
    tot_test, tot_acc = 0.0, 0.0
    
    ###################
    # train the model #
    ###################
    if (replay): replay_iter = iter(replay_train_loader)

    for data, target in train_loader:

        data = data.to(device)
        target = target.to(device)
        
        if (replay):
            # load replay data
            try: 
                replay_data, replay_target = next(replay_iter)
            except StopIteration:
                replay_iter = iter(replay_train_loader)
                replay_data, replay_target = next(replay_iter)
            #replay_data, replay_target = replay_data.to(device), replay_target.to(device)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        pred, recon, mu, logvar = model(data)
        if (replay):
            replay_pred, replay_recon, replay_mu, replay_logvar = model(replay_data)
        # calculate the loss
        loss_tot, loss_class, loss_rec = loss_function_vib(recon, data,  mu, logvar, pred, target, beta_1, beta_2, replay, model_class = True)

        # Exception: Crash
        if (loss_tot is None): return None, None, None, None

        if (replay):
            replay_loss_tot, replay_loss_class, replay_loss_rec = loss_function_vib(replay_recon, replay_data,  replay_mu, replay_logvar, replay_pred, replay_target, beta_1, beta_2, True, model_class=True)
            loss = loss_tot * (1-replay_weight) + replay_loss_tot * replay_weight
            class_loss += (loss_class.item() * (1 - replay_weight) + replay_loss_class.item() * replay_weight)
            rec_loss += (loss_rec.item() * (1 - replay_weight) + replay_loss_rec.item() * replay_weight)
        else: 
            loss = loss_tot
            class_loss = loss_class.item()
            rec_loss = loss_rec.item()
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
 
        # perform a single optimization step (parameter update)
        optimizer.step()
        #update running training loss
        train_loss += loss_tot.item()*(data.size(0))
     #   print(f'loss: {loss}')

        # Calculate accuracy of the batch
        pred = torch.nn.functional.softmax(pred, dim=1)
        pred = torch.max(pred, dim=1)[1]
        acc = pred.eq(target).sum().item()
        tot_acc += acc
        tot_test += data.size()[0]
        
    
    return train_loss, class_loss, rec_loss, tot_acc / tot_test


def train_vib(model, train_loader, optimizer, device, beta_1, beta_2):


    # if (replay != "no_replay" and n_tasks_so_far > 1): 
    #     return train_with_replay_vib(model, train_loader, replay_train_loader, optimizer, device, beta_1, beta_2, n_tasks_so_far, replay_weight)


    model.train()
    train_loss, class_loss, rec_loss = 0.0, 0.0, 0.0
    tot_test, tot_acc = 0.0, 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        pred, recon, mu, logvar = model(data)
        # Calculate losses. Loss_tot is used for backpropagation, loss-class and loss_rec for monitoring. in case beta_1 >0, loss_rec is actually the loss for prior match (KLD)
        loss_tot, loss_class, loss_rec = loss_function_vib(recon, data,  mu, logvar, pred, target, beta_1, beta_2)

        # Exception: Crash
        if (loss_tot is None): return None, None, None, None
        
        loss_tot.backward()
        train_loss += loss_tot.item()
        class_loss += loss_class.item()
        rec_loss += loss_rec.item()
        optimizer.step()

        # Calculate accuracy of the batch
        pred = torch.nn.functional.softmax(pred, dim=1)
        pred = torch.max(pred, dim=1)[1]
        acc = pred.eq(target).sum().item()
        tot_acc += acc
        tot_test += data.size()[0]

    return train_loss, class_loss, rec_loss, tot_acc / tot_test


def visualize_model(trainer, model, decoder_class, decoder_rec, b1, b2):

    for j, (data, target) in enumerate(trainer):
        data, target = data.to(config.device), target.to(config.device)
        pred, recon, _, _ = model(data)
        pred = torch.max(pred, dim=1)[1]
        recon = recon.detach().numpy()
        print(recon.shape)
        #recon = recon.view(28,28,1)
        fig = plt.figure(figsize=(25, 8))
        for idx in numpy.arange(40):
                    ax = fig.add_subplot(4, 10, idx+1, xticks=[], yticks=[])
                    ax.imshow(recon[idx].reshape(28,28), cmap='gray')
                    ax.set_title(str(target[idx].item())+ '___'+ str(pred[idx].item()))
        plt.savefig('results/vib/'+ path + '/beta_1_' + str(b1) + '_beta_2_' + str(b2) + '_sample.png')
        plt.close()
        break


    # show one set of generated images
    with torch.no_grad():
        sample = torch.randn(40, 100)
        sample_pred = decoder_class(sample)
        sample_pred = torch.max(sample_pred, dim=1)[1]
        sample = decoder_rec(sample)
                #recon = recon.view(28,28,1)
        fig = plt.figure(figsize=(25, 8))
        for idx in numpy.arange(40):
                    ax = fig.add_subplot(4, 10, idx+1, xticks=[], yticks=[])
                    ax.imshow(sample[idx].reshape(28,28), cmap='gray')
                    ax.set_title(str(sample_pred[idx].item()))
        plt.savefig('results/vib/'+ path + '/beta_1_' + str(b1) + '_beta_2_' + str(b2) + 'generated.png')
        plt.close()


 




