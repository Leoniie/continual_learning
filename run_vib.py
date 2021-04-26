""" Run the ProbMLP model in a continual task setting """
import argparse
import json
import logging
import os
import sys
import torchvision
import datetime
import matplotlib.pyplot as plt
import numpy
import csv


import torch

from lib import config, utils, model, train, replay, vae, vib_model


def load_default_config(task):
    """
    Load default parameter configuration from file.
    Args:
        tasks: String with the task name

    Returns:
        Dictionary of default parameters for the given task
    """

    if task == "perm_mnist":
        default_config = "etc/perm_mnist.json"
    elif task == "split_mnist":
        default_config = "etc/split_mnist.json"
    elif task == "split_mnist_vib":
        default_config = "etc/split_mnist_vib.json"
    elif task == "perm_mnist_vib":
        default_config = "etc/perm_mnist_vib.json"
    else:
        raise ValueError("Task \"{}\" not defined.".format(task))

    with open(default_config) as config_json_file:
        cfg = json.load(config_json_file)

    return cfg

    
def parse_shell_args(args):
    """
    Parse shell arguments for this script and return as dictionary
    """
    parser = argparse.ArgumentParser(description="Simple Replay")

    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS,
                        help="Size of mini batches during training.")
    parser.add_argument("--enc_dimensions", type=int, nargs="+",
                        default=argparse.SUPPRESS, help="Dimensions of the encoder network.")
    parser.add_argument("--dec_class_dimensions", type=int, nargs="+",
                        default=argparse.SUPPRESS, help="Dimensions of the classification decoder network.")
    parser.add_argument("--dec_rec_dimensions", type=int, nargs="+",
                        default=argparse.SUPPRESS, help="Dimensions of the reconstruction decoder network.")
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS,
                        help="Number of epochs to train.")
    parser.add_argument("--learning_rate", type=float, default=argparse.SUPPRESS,
                        help="Learning rate of the optimizer.")
    parser.add_argument("--beta_1", type=float, default=argparse.SUPPRESS,
                        help="Factor for the reconstruction error (if negative)/KL Divergence (if positive).")
    parser.add_argument("--beta_2", type=float, default=argparse.SUPPRESS,
                        help="Factor for the KL Divergence, must be positive.")
    parser.add_argument("--log_dir", type=str, default="",
                        help="Subdirectory within ./log/ where to store logs.")
    parser.add_argument("--nonlinearity", choices=["leaky_relu", "relu", "sigmoid", "tanh", "elu"],
                        default=argparse.SUPPRESS, help="Nonlinearity between network layers.")
    parser.add_argument("--optimizer", choices=["adam", "adagrad", "sgd"],
                        default=argparse.SUPPRESS, help="Optimizer used to train the model.")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Random seed for pytorch")
    parser.add_argument("--task", choices=["perm_mnist_vib", "split_mnist_vib"],
                        default="perm_mnist_vib", help="Continual task to be solved.")
    parser.add_argument("--weight_decay", type=float, default=argparse.SUPPRESS,
                        help="Weight decay (~l2-regularization strength) of the optimizer.") 
                        

    # Parse arguments to dictiodevicnary
    return vars(parser.parse_args(args))


def run_vib(cfg, path):

    os.mkdir('results/vib/' + path)
    
    # Initialized seed if specified (might slow down the model)
    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])


    # Load specified dataset
    train_loaders, test_loaders = utils.create_dataloader(cfg["dataset"]["name"], **cfg["dataset"]["kwargs"])

    # Initialize nonlinearity used as activation function
    nonlinearity = utils.create_nonlinearity(cfg['nonlinearity'])
 
    logging.info("Start experiment with parametrization:\n{}".format(json.dumps(cfg, indent=4)))
   
    # load first train and test data
    trainer = train_loaders[0]
    tester = test_loaders[0]
    replay_train_loader = None

    task = 0
    beta_1 = cfg['beta_1']
    beta_2 = cfg['beta_2']

    logging.info("Starting training of task {}".format(task + 1))

    beta_1_range = [-1, -0.1, -0.01, -0.001, -0.0001, 0, 0.0001, 0.001, 0.01, 0.1, 1]
    beta_2_range = [0, 0.0001, 0.001, 0.01, 0.1, 1]

    # Create a dictionary of the accuracies depending on different beta configurations
    accuracies = numpy.zeros((len(beta_1_range), len(beta_2_range)))
    best_epoch = numpy.zeros((len(beta_1_range), len(beta_2_range))) 

    for i, b1 in enumerate(beta_1_range):
        for j, b2 in enumerate(beta_2_range):

            # CREATE VIB MODEL

            logging.info("Start training with beta_1: {:.4f}, beta_2: {:.4f}".format(b1, b2))

            # Initialize Encoder
            encoder = vib_model.ENCODER(cfg['enc_dimensions'], nonlinearity)
            # Classification decoder
            decoder_class = vib_model.DECODER(cfg['dec_class_dimensions'], nonlinearity)
            # Reconstruction decoder
            decoder_rec = vib_model.DECODER(cfg['dec_rec_dimensions'], nonlinearity)
            # Initialize VIB model
            model = vib_model.VIBModel(encoder, decoder_class, decoder_rec)
            # OPTIMIZER --> pass list of  VIB model parameters to one optimizer
            optimizer = utils.create_optimizer(cfg['optimizer'], model, lr = cfg['learning_rate'])

            # TRAINING

            for epoch in range(cfg['epochs']):

                # Train model
                train_loss, class_loss, rec_loss, train_acc = vib_model.train_vib(model, trainer, replay_train_loader, optimizer, config.device, b1, b2)
                test_acc = vib_model.accuracy(model, tester, config.device)

                # update accuracies and store number of best epoch
                if (test_acc > accuracies[i][j]): 
                    accuracies[i][j] = test_acc
                    best_epoch[i][j] = epoch

                # # Logging
                logging.info("epoch {}: train_acc: {:.4f} \t test_acc: {:.4f}".format(epoch, train_acc, test_acc))
                config.writer.add_scalars('task{}/accuracy'.format(task + 1)+'_beta_1_{}'.format(b1)+'_beta_2_{}'.format(b2), {'train': train_acc, 'test': test_acc}, epoch)
                config.writer.add_scalar('task{}/train_loss'.format(task + 1)+'_beta_1_{}'.format(b1)+'_beta_2_{}'.format(b2), train_loss, epoch)
                config.writer.add_scalar('task{}/class_loss'.format(task + 1)+'_beta_1_{}'.format(b1)+'_beta_2_{}'.format(b2), class_loss, epoch)
                config.writer.add_scalar('task{}/rec_loss'.format(task + 1)+'_beta_1_{}'.format(b1)+'_beta_2_{}'.format(b2), rec_loss, epoch)

            # show how the model performs on one set of inputs and what kind of images it generates, store the images
            vib_model.visualize_model(trainer, model, decoder_class, decoder_rec, b1, b2)

            
    return beta_1_range, beta_2_range, accuracies, best_epoch
       



if __name__ == '__main__':
    
    # Parse the shell arguments as input configuration
    user_config = parse_shell_args(sys.argv[1:])


    # Load default parameter configuration from file
    cfg = load_default_config(user_config["task"])

    # Overwrite default parameters with user configuration where specified
    cfg.update(user_config)

    # Setup global logger and logging directory
    path = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    config.setup_logging( path + "_vib_"+"" + cfg["dataset"]["name"]+ "_vib", dir=cfg['log_dir'])

    # Run the script using the created parameter configuration
    beta_1_range, beta_2_range, accuracies, best_epoch = run_vib(cfg, path)

    # Store results (as csv?)

   # utils.list_to_csv(task_accuracies, os.path.join(config.log_dir, "task_accuracies.csv"))
    utils.list_to_csv(beta_1_range, os.path.join(config.log_dir, "beta_1_range.csv"))
    utils.list_to_csv(beta_2_range, os.path.join(config.log_dir, "beta_2_range.csv"))
    numpy.savetxt(os.path.join(config.log_dir, "accuracies.csv"), accuracies, fmt ='%f',delimiter = ',')
    numpy.savetxt(os.path.join(config.log_dir, "best_epoch.csv"), best_epoch,fmt =  '%i', delimiter = ',')

    print('done')