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

from itertools import product


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
    parser.add_argument("--model", type=str, default="comparison_model",
                        help="Model to train. Default: VIB Model. Alternative: Comparison model to produce a baseline")
    parser.add_argument("--cl_method", type=str, default="generative_replay",
                        help="Continual learning method to use, default: none. Options: no replay, simple replay and generative replay")
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS,
                        help="Number of epochs to train.")
    parser.add_argument("--freeze", type=bool, default=argparse.SUPPRESS,
                        help="Freeze the encoder during training of the second task")
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
    parser.add_argument("--n_replay", type=int,
                        default=argparse.SUPPRESS, help="Number of samples for each task to be replayed, if replay method applied.")
    parser.add_argument("--n_bottleneck", type=int,
                        default=argparse.SUPPRESS, help="Number of samples for each task to be replayed, if replay method applied.")
    parser.add_argument("--optimizer", choices=["adam", "adagrad", "sgd"],
                        default=argparse.SUPPRESS, help="Optimizer used to train the model.")
    parser.add_argument("--replay_weight", type=float,
                        default=argparse.SUPPRESS, help="Weight of replay loss in combined loss when training with replay. Default is 0.5. The weight of the training loss of the new data will be: 1 - replay_weight")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Random seed for pytorch")
    parser.add_argument("--task", choices=["perm_mnist_vib", "split_mnist_vib"],
                        default="perm_mnist_vib", help="Continual task to be solved.")
    parser.add_argument("--weight_decay", type=float, default=argparse.SUPPRESS,
                        help="Weight decay (~l2-regularization strength) of the optimizer.") 



    # Parse arguments to dictiodevicnary
    return vars(parser.parse_args(args))


def run_replay_vib(cfg, path, logdir, run_id):

    # Initialized seed if specified (might slow down the model)
    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])


    # Load specified dataset
    train_loaders, test_loaders, permutations = utils.create_dataloader(cfg["dataset"]["name"], **cfg["dataset"]["kwargs"])


    # Number of tasks fixed to 2, otherwise we can't use the permuations to reconstruct the generated images
    num_tasks = 2
    train_loaders = train_loaders[:2]
    test_loaders = test_loaders[:2]
    trainer_1 = train_loaders[0]
    tester_1 = test_loaders[0]
    trainer_2 = train_loaders[1]
    tester_2 = test_loaders[1]

    # Calculate inverse permutation for visulazation of perm_mnist
    if (cfg['task'] == 'perm_mnist_vib'): 
        inverse = utils.inverse_permutation(permutations[0])
    else:
        inverse = None
    
        
    # CREATE VIB MODEL
    # append dimension of bottleneck units to dimension speciications
    cfg['enc_dimensions'][-1] = cfg['n_bottleneck']
    cfg['dec_class_dimensions'][0] = cfg['n_bottleneck']
    cfg['dec_rec_dimensions'][0] = cfg['n_bottleneck']

    # Initialize nonlinearity used as activation function
    nonlinearity = utils.create_nonlinearity(cfg['nonlinearity'])
    # Initialize Encoder
    encoder = vib_model.ENCODER(cfg['enc_dimensions'], nonlinearity).to(config.device)
    # Classification decoder
    decoder_class = vib_model.DECODER(cfg['dec_class_dimensions'], nonlinearity).to(config.device)

    # define model
    if (cfg['model']== "vib_model"):
        # Define reconstruction deocder 
        decoder_rec = vib_model.DECODER(cfg['dec_rec_dimensions'], nonlinearity).to(config.device)
        model = vib_model.VIBModel(encoder, decoder_class, decoder_rec).to(config.device)
        # define optimizer
        optimizer = utils.create_optimizer(cfg['optimizer'], model, lr = cfg['learning_rate'])
    elif (cfg['model'] == "comparison_model"):    
        # define classification model
        model_class = vib_model.VIBModel_class(encoder, decoder_class).to(config.device)
        optimizer_class = utils.create_optimizer(cfg['optimizer'], model_class, lr = cfg['learning_rate'])

        if (cfg['cl_method'] == "generative_replay"):
            n_hidden = cfg['dec_rec_dimensions'][1]
            model_VAE = vae.VAE(cfg['n_bottleneck'], n_hidden).to(config.device)
            optimizer_VAE = utils.create_optimizer(cfg['optimizer'], model_VAE, lr = cfg['learning_rate'])
        else: model_VAE = None
    
    else:
        raise ValueError("Model not defined.")


    logging.info("Start experiment with parametrization:\n{}".format(json.dumps(cfg, indent=4)))
   
     # if replay method applied,create a list that stores a randomly sampled batch for each task
    replay_train_data, replay_test_data = [],[] # a list of replay data needs to be stored for simple replay, when the same replay data is used for a task
    replay_train_loader, replay_test_loader = None, None


    # define betas
    beta_1 = cfg['beta_1']
    beta_2 = cfg['beta_2']

    reconstruction_loss = 0 # Initialize reconstruction for the case of simple replay, where we don't actually compute it for the compare model

    ''' ========================= TRAINING ON TASK 1 ==============================='''
    task = 0

    logging.info("Starting training of task {}".format(task + 1))

    # TRAINING
    for epoch in range(cfg['epochs']):

        # Train model
        if (cfg['model']=="vib_model"):
            train_loss, class_loss, rec_loss, train_acc = vib_model.train_vib(model, trainer_1, optimizer, config.device, beta_1, beta_2)
            if train_loss is None:
                logging.info("Configuration lead to crash")
                return list((0,0))
            test_acc = vib_model.accuracy(model, tester_1, config.device)
            reconstruction_loss = rec_loss # We want to report this loss to compare the reconstruction data
        else:
            # We are training the comparison model
            # Train classifier
            train_loss, class_loss, rec_loss, train_acc = vib_model.train_model_class(model_class, trainer_1, replay_train_loader, optimizer_class, config.device, beta_1, beta_2, cfg['replay_weight'], replay = False)
            if train_loss is None:
                logging.info("Configuration lead to crash")
                return list((0,0))
            test_acc = vib_model.accuracy(model_class, tester_1, config.device)
        

                # # Logging
        utils.update(epoch, train_acc, test_acc, task, train_loss, class_loss, rec_loss)

    
    if ((cfg['model'] == "comparison_model") and (cfg['cl_method']=='generative_replay')):
        # Train VAE
        print(f'Training VAE on task: {task+1}')
        for vae_epoch in range(cfg['epochs']):
            vae_loss = vae.train(model_VAE, trainer_1, replay_train_loader, optimizer_VAE, config.device)
            vae.test(model_VAE, tester_1, config.device)
        reconstruction_loss = vae_loss


    

    # images, labels = next(iter(trainer_1))
    # grid = torchvision.utils.make_grid(images)
    # config.writer.add_image("images", grid)
    # config.writer.add_graph(model, images)
   
    # Create dictoinary of accuracies on all tasks
    if (cfg['model']=='vib_model'):
        task_accuracies = utils.accuracies(model, tester_1, config.device, task, test_loaders)
    else:
        task_accuracies = utils.accuracies(model_class, tester_1, config.device, task, test_loaders)

    acc_1 = task_accuracies['task1'] # accuracy on first task 




    '''========================== UPDATE THE REPLAY BUFFER========================='''

    # option "no replay" is realized by setting the replay weight to 0
    # the configuration "mult_mlp" is irrelevant here since we definitely only have one mlp
    if (cfg['model'] == "vib_model"):
        replay_train_loader, replay_train_data, replay_test_loader, replay_test_data = replay.update_replay_buffer(trainer_1, tester_1, replay_train_loader, replay_test_loader, replay_train_data, replay_test_data, decoder_rec, cfg['cl_method'], cfg['n_replay'], cfg['n_bottleneck'], cfg["dataset"]["kwargs"]['batch_size'], config.device, decoder_class, task, cfg['mult_mlp'], vib =  True)
    else:
        replay_train_loader, replay_train_data, replay_test_loader, replay_test_data = replay.update_replay_buffer(trainer_1, tester_1, replay_train_loader, replay_test_loader, replay_train_data, replay_test_data, model_VAE, cfg['cl_method'], cfg['n_replay'], cfg['n_bottleneck'], cfg["dataset"]["kwargs"]['batch_size'], config.device, model_class, task, cfg['mult_mlp'], vib =  False)

    
    # visualize one batch of training data and store in results
    utils.visualize(config.writer, run_id, replay_train_data, cfg['cl_method'], cfg['task'], logdir, inverse)
   # std_dev = utils.standard_deviation(replay_train_data, cfg['cl_method'])


    
    ''' ============================= TRAINING ON TASK 2 ========================== '''
    task = 1

    if (cfg['model'] == "vib_model"): 
        # define new optimizer
        model_class = vib_model.VIBModel_class(encoder, decoder_class).to(config.device)
        # define optimizer -  Two options: Either freeze encoder or train again
        if (cfg['freeze'] is True):
            optimizer_class = utils.create_optimizer(cfg['optimizer'], decoder_class, lr = cfg['learning_rate'])
        else:
            optimizer_class = utils.create_optimizer(cfg['optimizer'], model_class, lr = cfg['learning_rate'])
    else:
        # Freeze option can also be activated for comparison model
        if (cfg['freeze'] is True):
            optimizer_class = utils.create_optimizer(cfg['optimizer'], decoder_class, lr = cfg['learning_rate'])

    
    # TRAINING
    logging.info("Starting training of task {}".format(task + 1))
    for epoch in range(cfg['epochs']):

        # Train model
        train_loss, class_loss, rec_loss, train_acc = vib_model.train_model_class(model_class, trainer_2, replay_train_loader, optimizer_class, config.device, beta_1, beta_2, cfg['replay_weight'], replay = True)
        if train_loss is None:
            logging.info("Configuration lead to crash")
            return list((0,0))
        test_acc = vib_model.accuracy(model_class, tester_2, config.device)

        # # Logging
        utils.update(epoch, train_acc, test_acc, task, train_loss, class_loss, rec_loss)

        
        # Create dictoinary of accuracies on all tasks

    task_accuracies = {
        "task{}".format(task + 1): vib_model.accuracy(model_class, tester, config.device)
        for task, tester in enumerate(test_loaders)
    }

    acc_1_forget = task_accuracies['task1'] # accuracy on second task
    forgetting = acc_1 - acc_1_forget # forgetting of first task

    # Compute the mean accuracy over all tasks
    mean_accuracy = torch.mean(torch.Tensor(list(task_accuracies.values())[:task+1])) # compute mean accuracy over all tasks seen so far

    logging.info("Task accuracies: {}".format(json.dumps(task_accuracies, indent=4, sort_keys=True)))
    logging.info("Mean task accuracy: {:.4f}".format(mean_accuracy))
    config.writer.add_scalar('continual/mean_accuracy', mean_accuracy, task + 1)
    config.writer.add_scalars('continual/task_accuracies', task_accuracies, task + 1)

    # Hyperparameter tracking
    config.writer.add_hparams(
            {"model": cfg['model'], "lr": cfg['learning_rate'], "beta_1": cfg['beta_1'], "beta_2": cfg['beta_2'], "lr": cfg['learning_rate'], "beta_1": cfg['beta_1'], "beta_2": cfg['beta_2'], "freeze": cfg['freeze'], "n_bottleneck": cfg['n_bottleneck'], "n_replay" : cfg['n_replay']}, 
            {
                "accuracy task 1 round 1": acc_1,
                "accuracy task 1 round 2": acc_1_forget,
                "accuracy task 2 round 2": task_accuracies['task2'],
                "mean accuracy round 2": mean_accuracy,
                "forgetting": forgetting,
                "rec_loss": rec_loss
            },
        )

    return [task_accuracies[key] for key in sorted(task_accuracies)], forgetting, reconstruction_loss
       


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

    
    #parameters = dict(learning_rate = [0.0001, 0.001, 0.005, 0.01], beta_1 = [-100, -10, -1, -0.1, -0.01, 0], beta_2 = [0,0.01, 0.1, 1,10,100], freeze = [False], n_bottleneck = [10, 100, 1000], n_replay = [1000], model = ["vib_model", "comparison_model"], optimizer = ["adam"])

    parameters = dict(learning_rate = [0.0001], beta_1 = [0], beta_2 = [0,0.01, 0.1, 1,10,100], freeze = [False], n_bottleneck = [10, 100, 1000], n_replay = [1000], model = ["vib_model", "comparison_model"], optimizer = ["adam"])


    # RUN THIS NEXT: Everything the same, except for more beta_2: [1, 10, 100]

    # maybe run with other optimizers: adam, adagrad?

    #parameters = dict(learning_rate = [0.005], beta_1 = [-0.01], beta_2 = [ 0.01], freeze = [False], n_bottleneck = [50], n_replay = [1000], model = ["vib_model"], optimizer = ["adam"])



    param_values = [v for v in parameters.values()]

 
    with open(os.path.join(config.log_dir,'run_configurations.csv'), 'w',  newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["Run_id", "learning_rate", "beta_1", "beta_2", "model", "n_bottlneck", "n_replay", "optimizer", "acc_1", "acc_2", "forgetting", "rec_loss"])

        for run_id, (learning_rate, beta_1, beta_2, freeze, n_bottleneck, n_replay, model, optimizer) in enumerate(product(*param_values)):

            logging.info(f"Training run_id {run_id} with learning_rate {learning_rate}, beta_1 {beta_1},beta_2 {beta_2} and freeze {freeze}")
            cfg['learning_rate'] = learning_rate
            cfg['beta_1'] = beta_1
            cfg['beta_2'] = beta_2
            cfg['freeze'] = freeze
            cfg['model'] = model
            cfg['n_bottleneck'] = n_bottleneck
            cfg['n_replay'] = n_replay
            cfg['optimizer'] = optimizer
        # Run the script using the created parameter configuration
            task_accuracies, forgetting, reconstruction_loss = run_replay_vib(cfg, path, config.log_dir, run_id)
            # Store results (as csv?)
            #utils.list_to_csv(task_accuracies, os.path.join(config.log_dir, "task_accuracies_run_" + str(run_id) + ".csv"))

            writer.writerow([run_id,cfg['learning_rate'],cfg['beta_1'], cfg['beta_2'], cfg['freeze'], cfg['model'], cfg['n_bottleneck'], cfg['n_replay'], cfg['optimizer'],task_accuracies[0], task_accuracies[1], forgetting, reconstruction_loss])
    
    print('done')