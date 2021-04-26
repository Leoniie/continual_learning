from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision
import matplotlib.pyplot as plt
import numpy


from lib import config
#import config


class VAE(nn.Module):
    '''
    VAE as given in the pytorch github example. This VAE is NOT symmetric, contrary to the vandeVen impelementation

    Args:
        n_hidden: number of nodes in the hidden layers
        n_gaussian: number of nodes in the innermost random gaussian layer 
                (number of dimenions to generate random samples with the decoder)
    '''
    def __init__(self, n_gaussian, n_hidden):
        super(VAE, self).__init__()
        
        self.n_gaussian = n_gaussian
        self.n_hidden = n_hidden
        self.fc1 = nn.Linear(784, n_hidden)
        self.fc2 =nn.Linear(n_hidden, n_hidden)
        self.fc31 = nn.Linear(n_hidden, n_gaussian)
        self.fc32 = nn.Linear(n_hidden, n_gaussian)
        self.fc4 = nn.Linear(n_gaussian, n_hidden)
        self.fc5 = nn.Linear(n_hidden, n_hidden)
        self.fc6 = nn.Linear(n_hidden, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return torch.sigmoid(self.fc6(h5))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar




# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(model,train_loader, replay_train_loader, optimizer, device, single_vae=False):

    if ((not(replay_train_loader==None)) and single_vae==True): return train_with_replay(model,train_loader, replay_train_loader, optimizer, device)

    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss


def train_with_replay (model,train_loader, replay_train_loader, optimizer, device):
    model.train()
    train_loss = 0

    replay_iter = iter(replay_train_loader)

    for batch_idx, (data, _) in enumerate(train_loader):

        # task data
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

        # load replay data
        try: 
            replay_data, _ = next(replay_iter)
        except StopIteration:
            replay_iter = iter(replay_train_loader)
            replay_data, _ = next(replay_iter)
        
        replay_data = replay_data.to(device)
        replay_recon_batch, replay_mu, replay_logvar = model(replay_data)


        loss_task = loss_function(recon_batch, data, mu, logvar)
        loss_replay = loss_function(replay_recon_batch, replay_data, replay_mu, replay_logvar)

        loss = loss_task * (1/2) + loss_replay * (1/2)
        loss.backward()
        train_loss += loss_task.item()
        optimizer.step()
    
    return train_loss

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            #     save_image(comparison.cpu(),
            #              'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    #print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    n_inner = 20 # inner dimensions of the VAE
    model = VAE(100,400).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    for j, (data, target) in enumerate(train_loader):
        data = data.to(config.device)
        fig = plt.figure(figsize=(25, 8))
        for idx in numpy.arange(40):
                    ax = fig.add_subplot(4, 20/2, idx+1, xticks=[], yticks=[])
                    ax.imshow(numpy.squeeze(data[idx]), cmap='gray')
                    ax.set_title(str(target[idx].item()))
        plt.savefig('results/vae_'+ str(j)  +'.png')
        break
    

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader,None, optimizer, device)
        test(model, test_loader,device)
        with torch.no_grad():
            sample = torch.randn(64, n_inner).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                        'results/sample_' + str(epoch) + '.png')