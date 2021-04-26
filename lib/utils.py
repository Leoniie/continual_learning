import csv
import json
import os

import torch
from torchvision import transforms


import matplotlib.pyplot as plt
import numpy
from scipy import ndimage

from lib import data, config, vib_model
from PIL import Image
from lib.data import _permute_tensor

import logging


def compute_frozen_probs(mask_list):
    """ Compute the ratio of weight probabilities exceeding the freezing threshold
    Args:
        mask_list (List[Tensor]): list of binary tensors determining if weight is frozen

    Returns:
        Scalar Tensor
    """
    num_frozen = sum(torch.sum(m) for m in mask_list)
    num_total = sum(torch.numel(m) for m in mask_list)

    return num_frozen.float() / num_total


def compute_mean_probs(probs_list):
    """ Compute the mean weight probabilities
    Args:
        probs_list (List[Tensor]): list of tensors containing probabilities

    Returns:
        Scalar Tensor
    """
    # Concatenate all tensors into a single 1D tensor
    probs_cat = torch.cat([p.view(-1) for p in probs_list])

    return torch.mean(probs_cat)


def create_dataloader(name, **kwargs):
    """
    Return test and train dataloader given name of dataset
    """
    permutations = None

    if name == "mnist":
        train_loader, test_loader = data.create_mnist_loader(**kwargs)
        # Wrap in list to mimic a T=1 continual task
        train_loaders, test_loaders = [train_loader], [test_loader]
    elif name == "perm_mnist":
        train_loaders, test_loaders, permutations = data.create_perm_mnist_loader(**kwargs)
    elif name == "perm_fmnist":
        train_loaders, test_loaders = data.create_perm_fmnist_loader(**kwargs)
    elif name == "split_mnist":
        train_loaders, test_loaders = data.create_split_mnist_loader(**kwargs)
    elif name == "split_fmnist":
        train_loaders, test_loaders = data.create_split_fmnist_loader(**kwargs)
    else:
        raise ValueError("Dataset \"{}\" undefined".format(name))

    return train_loaders, test_loaders, permutations

def create_multitask_dataloader(name,**kwargs):
    if name == "split_mnist":
        train_loader, test_loader = data.create_multitask_split_mnist_loader(**kwargs)
    elif name == "perm_mnist":
        train_loader, test_loader = data.create_multitask_perm_mnist_loader(**kwargs)

    return train_loader, test_loader


def create_nonlinearity(name):
    """
    Return nonlinearity function given its name
    """
    if name == "leaky_relu":
        return torch.nn.functional.leaky_relu
    elif name == "relu":
        return torch.nn.functional.relu
    elif name == "sigmoid":
        return torch.sigmoid
    elif name == "tanh":
        return torch.nn.functional.tanh
    elif name == "elu":
        return torch.nn.functional.elu
    else:
        raise ValueError("Nonlinearity \"{}\" undefined".format(name))


def create_optimizer(name, model, **kwargs):
    """
    Return optimizer for the given model
    """
    if name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), **kwargs)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), **kwargs)
    else:
        raise ValueError("Optimizer \"{}\" undefined".format(name))


def list_to_csv(mylist, filepath):
    with open(filepath, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(mylist)


def save_dict_as_json(config, name, dir):
    """
    Store a dictionary as a json text file.
    """
    with open(os.path.join(dir, name + ".json"), 'w') as file:
        json.dump(config, file, sort_keys=True, indent=4)


def show_tensor(input):
    """
    Transform tensor into PIL object and show in separate window
    """
    image = transforms.functional.to_pil_image(input)
    image.show()

def get_one_hot_or_soft_target(target, device, num_classes):

    # TODO use T and make soft targets nicer?

    if len(target.shape) == 1:
        y = target.reshape(-1, 1).to(device)
        # One hot encoding buffer that you 
        # create out of the loop and just keep reusing
        y_onehot = torch.FloatTensor(len(target), 10).to(device)

        # In your for loop
        y_onehot.zero_()
        y_onehot = y_onehot.scatter_(1, y, 1).to(device)
        y_onehot = torch.nn.functional.one_hot(target, num_classes = num_classes)
    else:
        y_onehot = target

    return y_onehot

def soft_cross_entropy_v2(input, soft_target):
    return -(soft_target * torch.nn.functional.log_softmax(input, dim=1)).sum(dim=1)

def soft_cross_entropy(pred, soft_targets, num_classes, reduction = 'mean' , T=1):
    #print(f'Shape soft_targets: {len(soft_targets)}')
    soft_targets = get_one_hot_or_soft_target(soft_targets, config.device, num_classes)
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if (reduction == 'mean'):
        return (T**2) * torch.sum(soft_cross_entropy_v2(pred/T, soft_targets))/len(soft_targets)
    elif (reduction == 'sum'):
        return (T**2) * torch.sum(soft_cross_entropy_v2(pred/T, soft_targets))
    else:
        raise ValueError("Reduction \"{}\" not defined.".format(reduction))

    #return (T ** 2) * torch.sum(- torch.sum(soft_targets * logsoftmax(pred / T), 1))/len(soft_targets)



def plot_data(data, target):
    fig = plt.figure(figsize=(25, 4))
    labels = torch.max(target, dim=1)[1]
    for idx in numpy.arange(20):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        ax.imshow(numpy.squeeze(data[idx]), cmap='gray')
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        ax.set_title(str(labels[idx].item()))
    plt.savefig('show.png')


def visualize(writer, run_id, replay_data, cl_method, task, log_dir, inverse, n_images = 40 ,invert = True):

    if (cl_method == 'simple_replay'): replay_data =replay_data[0]

    torch.save(replay_data,  os.path.join(log_dir, "replay_data_" + cl_method + "_run_" + str(run_id) + ".pt"))

    fig = plt.figure(figsize=(25, 8))

    for j, (img, target) in enumerate(replay_data):
        if (j >= n_images): break

        img = img.cpu()

        if (task == 'perm_mnist_vib'):
            # Inverse permutation to reconstruct the image
            new = img.view(1, 784, -1)
            permuted = torch.zeros_like(new)
            for i in range(784):
                permuted[0][i] = new[0][inverse[i]]
            image = permuted.view(1, 28, -1)
        else:
            # split mnist does not need to be permuted
            image = img
        
        if (isinstance(target, int)): label = target
        else: label = torch.max(target, dim = 0)[1].item()

        ax = fig.add_subplot(4, 10, j+1, xticks=[], yticks=[])
        ax.imshow(numpy.squeeze(image), cmap='gray')
        ax.set_title(str(label))

   # writer.add_image("replay_sample_run_" + str(run_id), fig)
    plt.savefig(os.path.join(log_dir, "replay_data_sample_" + cl_method + "_run_" + str(run_id) + ".png"))

        
# def standard_deviation(replay_data, cl_method):
#     # Compute the average standard deviation of the replay data
#     if (cl_method == 'simple_replay'): replay_data =replay_data[0]
#     std_dev = 0
#     for j, (img, target) in enumerate(replay_data):
        
#         img = img.to(config.device)
#         std_dev = std_dev + ndimage.standard_deviation(img)
    
#     return std_dev/len(replay_data)
        
def loss_function_vib_recon(recon_x, x,  mu, logvar, pred, target, beta_1, beta_2):
    '''Computer the loss in case of substraction of beta_1 (reconstruction).
    
    Args:
        recon_x: image reconstructed by reconstruction decoder
        pred: label prediction made by classification decoder
        x: original input image
        mu, logvar: encoding after x passed through encoder, used for reparameterization
        beta_1, beta_2: factors used for VIB objective. Different loss function is called depending on whether beta_1 is substracted (reconstruction) or added (prior matching)
    
    Returns: (batch) loss
    '''
    criterion = torch.nn.CrossEntropyLoss()
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return criterion(pred,target) - beta_1 * BCE + beta_2 * KLD


def loss_function_vib_prior(mu, logvar, pred, target, beta_1, beta_2):
    '''Computer the loss in case of addition of beta_1 (prior matching).
    
    Args:
        x: original input image
        mu, logvar: encoding after x passed through encoder, used for reparameterization
        beta_1, beta_2: factors used for VIB objective. Different loss function is called depending on whether beta_1 is substracted (reconstruction) or added (prior matching)
    
    Returns: (batch) loss
    '''
    criterion = torch.nn.CrossEntropyLoss()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return criterion(pred,target)+ (beta_1 + beta_2) * KLD



def loss_function_vib(recon_x, pred, x, mu, logvar, beta_1, beta_2):
    '''Calculates the loss function for the VIB objecitve (to MINIMIZE). Different loss functions depending on whether beta_1 is substracted (reconstruction) or added (prior matching)
    
    Args:
        recon_x: image reconstructed by reconstruction decoder
        pred: label prediction made by classification decoder
        x: original input image
        mu, logvar: encoding after x passed through encoder, used for reparameterization
        beta_1, beta_2: factors used for VIB objective. Different loss function is called depending on whether beta_1 is substracted (reconstruction) or added (prior matching)
    
    Returns: 
        Calls loss function of the joint model (encoder, reconstrcution decoder, classification decoder)
    '''
    
    if (beta_1 <= 0): return loss_function_vib_recon(recon_x, pred, x, mu, logvar, beta_1, beta_2)
    else: return loss_function_vib_prior(x, mu, logvar, beta_1, beta_2)


def update(epoch, train_acc, test_acc, task, train_loss, class_loss, rec_loss):
    # Logging of the training results of vib replay
    logging.info("epoch {}: train_acc: {:.4f} \t test_acc: {:.4f}".format(epoch, train_acc, test_acc))
    config.writer.add_scalars('task{}/accuracy'.format(task + 1), {'train': train_acc, 'test': test_acc}, epoch)
    config.writer.add_scalar('task{}/train_loss'.format(task + 1), train_loss, epoch)
    config.writer.add_scalar('task{}/class_loss'.format(task + 1), class_loss, epoch)
    config.writer.add_scalar('task{}/rec_loss'.format(task + 1), rec_loss, epoch)

def accuracies(model, tester_1, device, task, test_loaders):
    task_accuracies = {
        "task{}".format(task + 1): vib_model.accuracy(model, tester, config.device)
        for task, tester in enumerate(test_loaders)
    }

    # Compute the mean accuracy over all tasks
    mean_accuracy = torch.mean(torch.Tensor(list(task_accuracies.values())[:task+1])) # compute mean accuracy over all tasks seen so far

    logging.info("Task accuracies: {}".format(json.dumps(task_accuracies, indent=4, sort_keys=True)))
    logging.info("Mean task accuracy: {:.4f}".format(mean_accuracy))
    config.writer.add_scalar('continual/mean_accuracy', mean_accuracy, task + 1)
    config.writer.add_scalars('continual/task_accuracies', task_accuracies, task + 1)

    return task_accuracies

def inverse_permutation(permutation):
    '''Computes the inverse permutation
    Args:
        permutation: array of permutation p
    Returns: 
        inverse: array of inverse  p^-1, type int
    '''

    inverse = numpy.zeros(len(permutation))
    for i in range(len(permutation)):
        inverse[permutation[i]] = i
    inverse = inverse.astype(int)

    return inverse
