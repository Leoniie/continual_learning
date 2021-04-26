import torchvision
from torchvision import datasets, transforms
import random
import torch
import logging

from lib import data, vae, train, config

# def train_model_on_replay_buffer(main_model, cfg, optimizer, replay_train_loaders, replay_test_loaders):
#     '''
#     Train main model on data that is in replay buffers. 

#     Args:
#         main_model: Main Model (Classifier, default: Multi-Layer Perceptron)
#         cfg: onfigurations as specified in .json file for task and input
#         replay_train_loaders, replay_test_loaders: dataloaders for that contain replay data of former tasks
#     '''

#     # Train on replay buffer
#     for replay_task, (replay_trainer,replay_tester) in enumerate(zip(replay_train_loaders, replay_test_loaders)):
#         logging.info("Starting replay of task {}".format(replay_task + 1))
#         # Train for specified amount of epochs
#         for replay_epoch in range(0, cfg['replay_epochs']):

#             replay_train_loss, replay_train_acc = train.train(main_model, replay_trainer, replay_tester, optimizer, cfg['criterion'])
#             replay_test_acc = train.eval(main_model, replay_tester, cfg['criterion'], cfg['dataset']['name'])
#             logging.info("Replay epoch {}: Replay_train_acc: {:.4f} \t Replay_test_acc: {:.4f}".format(replay_epoch, replay_train_acc, replay_test_acc))


def train_vae_with_replay(task, vae_model, trainer, tester, optimizer, device, replay_train_loader, replay_test_loader, vae_epochs):
    '''
    Train the VAE on the data for the given task and then train it on replay of former tasks as well. Use the same 
    data as the one that is used for replay on the classifier - i.e. data that was generated in former rounds.

    Args:
        task: task that the classifier and the vae are currently trained on, e.g. one instance of split-mnist: 2-3 or 8-9
        vae_model: vae_model (NOT the classifier model)
        trainer: dataloader for the training data for the task
        tester: dataloader for the test data for the task
        optimizer: vae optimizier as specified in configurations, default: ADAM
        device: config.device that is defined in main model (in case CUDA is available..)
        replay_train_loaders, replay_test_loaders: dataloader that contains replay data of former tasks
        vae_epochs: number of training epochs of the generative model (vae)

    '''
    logging.info("Starting to fit generator to task {}".format(task+1))
    for epoch in range(1, vae_epochs+1):

        if (task== 0): # only do replay if we have already seeen other tasks!
            vae_train_loss = vae.train(vae_model, trainer, optimizer, device)
            logging.info('====> VAE Epoch: {} Average loss: {:.4f}'.format(epoch, vae_train_loss / len(trainer.dataset)))
        else:
            vae_train_loss = vae.train_with_replay(vae_model, trainer, replay_train_loader, optimizer, device, task+1)
            logging.info('====> VAE Epoch: {} Average loss: {:.4f}'.format(epoch, vae_train_loss / len(trainer.dataset)))
        
        vae_test_loss = vae.test(vae_model, tester, device)
        logging.info('====> VAE  Test set loss: {:.4f}'.format(vae_test_loss))


def update_replay_buffer(trainer, tester, replay_train_loader, replay_test_loader, replay_train_data, replay_test_data, vae_models, cl_method, n_replay, vae_n_gaussian, batch_size, device, main_model, task, mult_mlp, vib = False):
    '''
    Update the replay buffer with a method depending on which type of replay we are using (simple or generative)

    Case simple = Add trainloader for sample data of current task to trainloader list
    Case generative = Replace trainloader with past examples with new samples created by the current vae and labeled by the main model

    Args:
        trainer, tester: dataloaders for training and testing data of current task
        replay_train_loaders, replay_test_loaders: dataloadersthat contain replay data of former tasks
        cfg: onfigurations as specified in .json file for task and input
        device: config.device that is defined in main model (in case CUDA is available..)
        main_model: Main Model (Classifier, default: Multi-Layer Perceptron). In case of VIB Model, the      main_model is the classification decoder
        vae_models: either list of VAEs or single VAE that is used to generate the replay samples. In case of the VIB model, the vae_model is just the reconstruction decoder. 
        epoch: Epoch
    Returns:
        replay_train_loader, replay_test_loader: Dataloaders for the train and test data
        replay_train_data, replay_test_data: List of data that is included in train and test loader (listed by task)
    '''


    if(cl_method=='simple_replay'): 
        logging.info("Update replay buffer for simple replay.")
        replay_train_loader, replay_train_data = update_replay_buffer_simple(replay_train_loader, replay_train_data, trainer.dataset, n_replay, batch_size, shuffle=True)
        replay_test_loader, replay_test_data = update_replay_buffer_simple(replay_test_loader, replay_test_data, tester.dataset, n_replay, batch_size, shuffle=True)
        logging.info("size of replay buffer: {}".format(len(replay_train_loader.dataset)))
    elif(cl_method=='generative_replay'):
        logging.info("Update replay buffer for generative replay.")
        with torch.no_grad():
            replay_train_loader, replay_train_data = update_replay_buffer_generative_mult_vae(main_model, vae_models, n_replay, vae_n_gaussian, batch_size, device, task, mult_mlp, vib)
            replay_test_loader, replay_test_data = update_replay_buffer_generative_mult_vae(main_model, vae_models,  n_replay, vae_n_gaussian, batch_size, device, task, mult_mlp, vib)    
    else: logging.info("CL method not defined.")

    return replay_train_loader, replay_train_data, replay_test_loader, replay_test_data



def update_replay_buffer_simple(replay_buffer, replay_data, dataset, n_replay, batch_size, shuffle):
    '''
    Add samples of the most recent task to the replay buffer 
    (i.e. train and test loaders with replay data for the tasks seen so far
    
    Args:
        replay_buffer: Replay buffer for replay samples of the tasks 0,...,i-1 in round i
        replay_data: list of data with replay samples for each tasks 0,...,91
        dataset: train/testset used in round i
        n_replay: number of replay samples to be chosen randomly from task i
        batch_size: same batch_size as for training of task
        shuffle: whether the dataloader should represent the replay data in a shuffled way. Default: False


    Returns:
        replay_dataloader: dataloader containing all the replay data
        replay_data: List of replay data extended by replay data for current task

    '''
    # create list of random indices
    subset_indices = random.sample(range(0, len(dataset)), n_replay)
    subset = torch.utils.data.Subset(dataset, subset_indices)
    replay_data.append(subset)

    # create dataloader from random subsets that includes all data
    loader_kwargs = data._get_loader_kwargs(num_workers=0)

 #   replay_data = replay_data.to(config.device)

    replay_dataloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(replay_data), batch_size, shuffle=shuffle, pin_memory = False, drop_last=False)
                                            # don't drop last since the subset might be a lot smaller than the bacth_size
    
    return replay_dataloader, replay_data
    



def update_replay_buffer_generative_full(replay_buffer_list,n_replay, n_inner, device, main_model, vae_model, batch_size, shuffle, is_train):
    '''
    Fill replay buffer with samples generated by the last version of the model and vae
    
    Args:
        replay_buffer_list: List of train/testloaders for replay samples of the tasks 0,...,i-1 in round i
        dataset: train/testset used in round i
        n_replay: number of replay samples to be chosen randomly from task i
        batch_size: same batch_size as for training of task
        shuffle: whether the dataloader should represent the replay data in a shuffled way. Default: False


    Returns:
        replay_dataloader: dataloader containing all the replay data
        replay_data: List of replay data extended by replay data for current task
    '''

    sample = torch.randn(n_replay, n_inner).to(device)
    sample = vae_model.decode(sample).cpu() 
    pred = main_model(sample)
    #pred = torch.max(pred, dim=1)[1]
    sample = sample.view(n_replay, 1, 28, 28) # change dimensions
    replay_data = []
    for item in range(0, len(sample)):
        #replay_data = torch.cat((replay_data, tuple((sample[item],pred[item]))),1)
        replay_data.append(tuple((sample[item],pred[item])))
        
 
    # create dataloader from random subset
    loader_kwargs = data._get_loader_kwargs(num_workers=0)
    replay_dataloader = torch.utils.data.DataLoader(replay_data, batch_size, shuffle=shuffle, pin_memory = False,
                                            drop_last=False)
        
    return replay_dataloader, replay_data


def update_replay_buffer_generative_mult_vae(main_model, vae_models, n_replay, n_gaussian, batch_size, device, task, mult_mlp, vib):
    '''
    Updata the replay buffer with samples for each task, use separate vaes that are trained
    on the specific tasks

    Args:
        main_model: Classifier (or in case of VIB Model: classification decoder)
        vae_model: List of vae models for the tasks seen so far, or just a single vae model (or in case of VIB Model: reconstruction decoder)
        replay_data_list: list of data with replay samples for each tasks 0,...,91
        n_replay: number of replay samples to be chosen randomly from task i
        n_inner: gaussian nodes of the vae model --> dimensions of space from which to sample encoding for generating random tasks
        batch_size: same batch_size as for training of task
        device: config.device that is defined in main model (in case CUDA is available..)
        vib: True if we are updating the replay buffer of the VIB model
    


    Returns:
        replay_buffer: replay buffer containing all the replay data
        replay_data_list: List of replay data for all tasks seen so far

    '''
    #replay_data = [] # List of datasets, one per sample
    replay_data_cat = [] # List of all the data
    soft = torch.nn.Softmax(dim=1)
    # go over all vae_models
    if (vib == False):
        if(type(vae_models) is not list): vae_models = [vae_models]
        for task_idx in range(len(vae_models)):
            sample = torch.randn(n_replay, n_gaussian).to(device)
            vae_model_task = vae_models[task_idx]
            # Generation of the sample data
                # Sample data is generated with vae models
            sample = vae_model_task.decode(sample) # maybe leave the 
            if not (mult_mlp):
                pred = main_model(sample)
            else:
                task_model = torch.load(f'models/intermediate_task_{task_idx}.pkl')
                pred = task_model(sample)
            if (type(pred) is tuple): pred = pred[0]  
            pred = soft(pred)
            generative_data = []
            sample = sample.view(n_replay, 1, 28, 28) # change dimensions
            for item in range(0, len(sample)):
                generative_data.append(tuple((sample[item],pred[item])))
                replay_data_cat.append(tuple((sample[item],pred[item])))
            
           # replay_data.append(generative_data)
        
    else:
        sample = torch.randn(n_replay, n_gaussian).to(device)
        # Sample data is generated with reconstruction decoder
        pred = main_model(sample)
        pred = soft(pred)
        sample = vae_models(sample)
        sample = torch.sigmoid(sample)
        #generative_data = []
        sample = sample.view(n_replay, 1, 28, 28) # change dimensions
        for item in range(0, len(sample)):
           # generative_data.append(tuple((sample[item],pred[item])))
            replay_data_cat.append(tuple((sample[item],pred[item])))
        
        #replay_data.append(generative_data)
 
    # create dataloader from random subsets that includes all data
  #  replay_data_cat = replay_data_cat.to(device)
    loader_kwargs = data._get_loader_kwargs(num_workers=0)
    replay_dataloader = torch.utils.data.DataLoader(replay_data_cat, batch_size, shuffle=True, pin_memory = False,
                                            drop_last=False)
                                            # don't drop last since the subset might be a lot smaller than the bacth_size
    
    return replay_dataloader, replay_data_cat