""" Run the ProbMLP model in a continual task setting """
import argparse
import json
import logging
import os
import sys
import torchvision
import datetime


import torch

from lib import config, utils, model, train, replay, vae

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
    parser.add_argument("--dimensions", type=int, nargs="+",
                        default=argparse.SUPPRESS, help="Dimensions of the neural network.")
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS,
                        help="Number of epochs to train.")
    parser.add_argument("--learning_rate", type=float, default=argparse.SUPPRESS,
                        help="Learning rate of the optimizer.")
    parser.add_argument("--log_dir", type=str, default="",
                        help="Subdirectory within ./log/ where to store logs.")
    parser.add_argument("--nonlinearity", choices=["leaky_relu", "relu", "sigmoid", "tanh"],
                        default=argparse.SUPPRESS, help="Nonlinearity between network layers.")
    parser.add_argument("--optimizer", choices=["adam", "adagrad", "sgd"],
                        default=argparse.SUPPRESS, help="Optimizer used to train the model.")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Random seed for pytorch")
    parser.add_argument("--task", choices=["perm_fmnist", "perm_mnist", "split_fmnist", "split_mnist"],
                        default="perm_mnist", help="Continual task to be solved.")
    parser.add_argument("--weight_decay", type=float, default=argparse.SUPPRESS,
                        help="Weight decay (~l2-regularization strength) of the optimizer.") 


    # Parse arguments to dictionary
    return vars(parser.parse_args(args))


def run_multitask(cfg):

    
    
    # Initialized seed if specified (might slow down the model)
    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])
    # Load specified dataset

    trainer, testers = utils.create_multitask_dataloader(cfg["dataset"]["name"], **cfg["dataset"]["kwargs"])

    # Initialize nonlinearity used as activation function
    nonlinearity = utils.create_nonlinearity(cfg['nonlinearity']) # TODO: is this needed here?

    # Initialize the model (need to(device) for adagrad)
    mlp = model.MLP(cfg['dimensions'], nonlinearity).to(config.device)
    
 
    # Define optimizer (may include l2 regularization via weight_decay)
    optimizer = utils.create_optimizer(cfg['optimizer'], mlp, lr=cfg['learning_rate'])                            
    logging.info("Start training with parametrization:\n{}".format(json.dumps(cfg, indent=4)))
    print("here")
    print(cfg['single_vae'])

    ''' Training'''

    replay_train_loader = None
    

    # Train for specified amount of epochs
    print("Start multitask training")

    for epoch in range(cfg['epochs']):
    

        train_loss, train_acc = train.train(mlp, trainer, optimizer, cfg['criterion'], cfg['cl_method'], replay_train_loader, 0)


        task_accuracies = {
            "task{}".format(task + 1): train.eval(mlp, testers[task], cfg['criterion'],10)
            for task, tester in enumerate(testers)
        }

        # Compute the mean accuracy over all tasks
        mean_accuracy = torch.mean(torch.Tensor(list(task_accuracies.values()))) # compute mean accuracy over all tasks seen so far

        # Logging
        logging.info("epoch {}: train_acc: {:.4f} \t mean_test_acc: {:.4f}".format(epoch, train_acc, mean_accuracy))
        config.writer.add_scalars('accuracy', {'train': train_acc, 'mean_test': mean_accuracy}, epoch)
        config.writer.add_scalar('train_loss', train_loss, epoch)

    logging.info("Task accuracies: {}".format(json.dumps(task_accuracies, indent=4, sort_keys=True)))
    logging.info("Mean task accuracy: {:.4f}".format(mean_accuracy))

    

    return [task_accuracies[key] for key in sorted(task_accuracies)]

    


# return


if __name__ == '__main__':
    
    # Parse the shell arguments as input configuration
    user_config = parse_shell_args(sys.argv[1:])


    # Load default parameter configuration from file
    cfg = load_default_config(user_config["task"])

    # Overwrite default parameters with user configuration where specified
    cfg.update(user_config)

    # Setup global logger and logging directory
    config.setup_logging(datetime.datetime.now().strftime("%Y%m%d_%H%M_")+ cfg["cl_method"]+"_"+"" + cfg["dataset"]["name"], dir=cfg['log_dir'])
    # Run the script using the created parameter configuration
    task_accuracies = run_multitask(cfg)

    # Store results (as csv?)

    utils.list_to_csv(task_accuracies, os.path.join(config.log_dir, "task_accuracies.csv"))

    print('done')