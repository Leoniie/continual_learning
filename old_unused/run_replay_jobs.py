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
    elif os.path.isfile(task):
        default_config = task
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
    parser.add_argument("--cl_method", type=str, default="no_replay",
                        help="Continual learning method to use, default: none. Other options : simple_replay, generative_replay")
    parser.add_argument("--mult_mlp", type=bool, default=False,
                        help="If mult_mlp==TRUE, old versions of the classifier will be stored and used to classify the generated samples")
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS,
                        help="Number of epochs to train.")
    parser.add_argument("--learning_rate", type=float, default=argparse.SUPPRESS,
                        help="Learning rate of the optimizer.")
    parser.add_argument("--log_dir", type=str, default="",
                        help="Subdirectory within ./log/ where to store logs.")
    parser.add_argument("--nonlinearity", choices=["leaky_relu", "relu", "sigmoid", "tanh"],
                        default=argparse.SUPPRESS, help="Nonlinearity between network layers.")
    parser.add_argument("--n_replay", type=int,
                        default=argparse.SUPPRESS, help="Number of samples for each task to be replayed, if replay method applied.")
    parser.add_argument("--optimizer", choices=["adam", "adagrad", "sgd"],
                        default=argparse.SUPPRESS, help="Optimizer used to train the model.")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Random seed for pytorch")
    parser.add_argument("--task", choices=["perm_fmnist", "perm_mnist", "split_fmnist", "split_mnist"],
                        default="perm_mnist", help="Continual task to be solved.")
    parser.add_argument("--weight_decay", type=float, default=argparse.SUPPRESS,
                        help="Weight decay (~l2-regularization strength) of the optimizer.") 
    parser.add_argument("--vae_batch_size", type=int, default=argparse.SUPPRESS, help="Size of minibatches during training of VAE in case of generative replay.")
    parser.add_argument("--vae_epochs", type=int, default=argparse.SUPPRESS, help="Number of epochs during training of VAE in case of generative replay.")
    parser.add_argument("--vae_n_gaussian", type=int, default=argparse.SUPPRESS, help="Number of inner random variables for VAE in case of generative replay.")
    parser.add_argument("--vae_learning_rate", type=float, default=argparse.SUPPRESS,
                        help="Learning rate of the VAE optimizer in case of generative replay.")


    # Parse arguments to dictiodevicnary
    return vars(parser.parse_args(args))


def run_simple_replay(cfg):
    
    # Initialized seed if specified (might slow down the model)
    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])


    # Load specified dataset

    train_loaders, test_loaders = utils.create_dataloader(cfg["dataset"]["name"], **cfg["dataset"]["kwargs"])

    # Initialize nonlinearity used as activation function
    nonlinearity = utils.create_nonlinearity(cfg['nonlinearity']) # TODO: is this needed here?

    # Initialize the model (need to(device) for adagrad)
    mlp = model.MLP(cfg['dimensions'], nonlinearity).to(config.device)
    
 
    # Define optimizer (may include l2 regularization via weight_decay)
    optimizer = utils.create_optimizer(cfg['optimizer'], mlp, lr=cfg['learning_rate'])                            
    logging.info("Start training with parametrization:\n{}".format(json.dumps(cfg, indent=4)))

    # if replay method applied,create a list that stores a randomly sampled batch for each task
    replay_train_data, replay_test_data = [],[]
    replay_train_loader, replay_test_loader = None, None
    
    
    ''' Training'''
    
    # Train each task (including simple replay of previous tasks)
    for task, (trainer, tester) in enumerate(zip(train_loaders, test_loaders)):

        #print train data
        # print('printing training data')
        # for j, (data, target) in enumerate(trainer):
        #             data = data.to(config.device)
        #             torchvision.utils.save_image(data.view(len(data), 1, 28, 28), 'results/traiing_sample_' + cfg['cl_method'] + '_task_' + str(task) + '.png')
        #             print(target)
        #             break
        
        logging.info("Starting training of task {}".format(task + 1))
        # Train for specified amount of epochs

        for epoch in range(cfg['epochs']):
            train_loss, train_acc = train.train(mlp, trainer, optimizer, cfg['criterion'], cfg['cl_method'], replay_train_loader, task+1)
            test_acc = train.eval(mlp, tester, cfg['criterion'], cfg['dataset']['name']) # test acc. only on current task

            # Logging
            logging.info("epoch {}: train_acc: {:.4f} \t test_acc: {:.4f}".format(epoch, train_acc, test_acc))
            config.writer.add_scalars('task{}/accuracy'.format(task + 1), {'train': train_acc, 'test': test_acc}, epoch)
            config.writer.add_scalar('task{}/train_loss'.format(task + 1), train_loss, epoch)

        # Create a dictionary of the accuracies on all tasks
        
        # NOTE: This yields an unsorted dictionary
        task_accuracies = {
            "task{}".format(task + 1): train.eval(mlp, tester, cfg['criterion'], cfg['dataset']['name'],10)
            for task, tester in enumerate(test_loaders)
        }

        # Compute the mean accuracy over all tasks
        mean_accuracy = torch.mean(torch.Tensor(list(task_accuracies.values())[:task+1])) # compute mean accuracy over all tasks seen so far

        logging.info("Task accuracies: {}".format(json.dumps(task_accuracies, indent=4, sort_keys=True)))
        logging.info("Mean task accuracy: {:.4f}".format(mean_accuracy))
        config.writer.add_scalar('continual/mean_accuracy', mean_accuracy, task + 1)
        config.writer.add_scalars('continual/task_accuracies', task_accuracies, task + 1)

        # If method is Generative Replay, train the VAE, too (also with replay)
        if(cfg['cl_method']=='generative_replay'):
            # Initialize new VAE for each task
            vae_model = vae.VAE(cfg['vae_n_gaussian'], cfg['vae_n_hidden']).to(config.device)
            vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=cfg['vae_lr'])
           
            print(f'Training VAE on task: {task}')
            for vae_epoch in range(cfg['vae_epochs']):
                vae.train(vae_model, trainer, vae_optimizer, config.device)
                vae.test(vae_model, tester, config.device)

                # vae_train_loss = vae.train(vae_model, trainer, vae_optimizer, config.device)
                # vae_test_loss = vae.test(vae_model, tester, config.device)
                #logging.info('====> VAE Epoch: {} Average loss: {:.4f}'.format(epoch, vae_train_loss / len(trainer.dataset)))
            
            # print generated data before saving
            with torch.no_grad():
                sample = torch.randn(64, cfg['vae_n_gaussian']).to(config.device)
                sample = vae_model.decode(sample)
                torchvision.utils.save_image(sample.view(64, 1, 28, 28),
                            'results/before_save_sample_' + str(epoch) + '.png')
            
            torch.save(vae_model, f'models/vae_task_{task}.pkl')
            
        
        if (cfg['mult_mlp']): 
            torch.save(mlp, f'models/intermediate_task_{task}.pkl') # store mlp version for each task

       # UPDATE REPLAY BUFFER

        if (not(cfg['cl_method']== 'no_replay')):
            replay_train_loader, replay_train_data, replay_test_loader, replay_test_data= replay.update_replay_buffer(trainer, tester, replay_train_loader, replay_test_loader, replay_train_data, replay_test_data, cfg['cl_method'], cfg['n_replay'], cfg['vae_n_gaussian'], cfg["dataset"]["kwargs"]['batch_size'], config.device, mlp, task, cfg['mult_mlp'])
            
            # visualize training data and store in results
            for i in range(len(replay_train_data)):
                for j, (data, target) in enumerate(replay_train_loader):
                    data = data.to(config.device)
                    torchvision.utils.save_image(data.view(len(data), 1, 28, 28), 'results/sample_' + cfg['cl_method'] + '_task_' + str(task) + '_epoch_' + str(epoch) + '.png')
                    #print(target)
                    break

                    
                    # print(f'target task {task}')
                    # print(target)
            

    return [task_accuracies[key] for key in sorted(task_accuracies)]

    


# return


if __name__ == '__main__':
    
    # Parse the shell arguments as input configuration
    user_config = parse_shell_args(sys.argv[1:])

   # job_folder = user_config["job_folder"]
   # for file in os.listdir(job_folder) and file.endswith("json"):
    # Load default parameter configuration from file
    cfg = load_default_config(user_config["task"])

    # Overwrite default parameters with user configuration where specified
    cfg.update(user_config)

    # Setup global logger and logging directory
    config.setup_logging(datetime.datetime.now().strftime("%Y%m%d_%H%M_")+ cfg["cl_method"]+"_"+"" + cfg["dataset"]["name"], dir=cfg['log_dir'])

    # Run the script using the created parameter configuration
    task_accuracies = run_simple_replay(cfg)

    # Store results (as csv?)

    utils.list_to_csv(task_accuracies, os.path.join(config.log_dir, "task_accuracies.csv"))

    print('done')