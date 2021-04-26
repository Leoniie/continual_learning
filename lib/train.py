
import torch

import numpy as np


from lib import config, utils





def train(model, train_loader, optimizer, crit, replay = "no_replay", replay_train_loader = None, n_tasks_so_far= 0, replay_weight=0.5):

    if (replay != "no_replay" and n_tasks_so_far > 1): return train_with_replay(model, train_loader, replay_train_loader, optimizer, crit, n_tasks_so_far, replay_weight)
    
    # Determine loss
    criterion = torch.nn.CrossEntropyLoss()

    correct_pred, train_loss = 0.0, 0.0
    
    ###################
    # train the model #
    ###################
    for data, target in train_loader:
        data, target = data.to(config.device), target.to(config.device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        #update running training loss
        train_loss += loss.item()*data.size(0)
        
        with torch.no_grad():
            pred = torch.max(output, dim=1)[1]
            correct_pred += torch.sum(torch.eq(pred, target)).item()

    # calculate average loss over an epoch, and average accuracy over an epoch
    train_loss = train_loss/len(train_loader.dataset)
    acc = correct_pred/len(train_loader.dataset)


    return train_loss, acc

def train_with_replay(model, train_loader, replay_train_loader, optimizer, crit, n_tasks_so_far, replay_weight):
    
    criterion = torch.nn.CrossEntropyLoss()
    correct_pred, train_loss = 0.0, 0.0
    
    ###################
    # train the model #
    ###################
    replay_iter = iter(replay_train_loader)

    for data, target in train_loader:
        # load replay data
        try: 
            replay_data, replay_target = next(replay_iter)
        except StopIteration:
            replay_iter = iter(replay_train_loader)
            replay_data, replay_target = next(replay_iter)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        replay_output = model(replay_data)
        # calculate the loss
        loss_pure = criterion(output, target)
        loss = loss_pure * (1-replay_weight) + utils.soft_cross_entropy(replay_output, replay_target, output.shape[1]) * replay_weight
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        #update running training loss
        train_loss += loss_pure.item()*(data.size(0))


        with torch.no_grad():
            pred = torch.max(output, dim=1)[1]
            correct_pred += torch.sum(torch.eq(pred, target)).item()
            # replay_pred = torch.max(replay_output, dim=1)[1]
            # correct_pred += torch.sum(torch.eq(replay_pred, replay_target)).item()
        
    # calculate average loss over an epoch, and average accuracy over an epoch
    train_loss = train_loss/(len(train_loader.dataset))
    acc = correct_pred/len(train_loader.dataset)


    return train_loss, acc


def eval(model, tester, criterion, num_samples= 10):

    '''Evaluate the model, calculate accuracy of the classes
    
    Args:
        model: here an MLP that takes as an input 28 x 28 pixel images
        tester: data loader for the testset
        criterion: loss function
        task_idx: index of the split mnist task, ranging from 0 to 4
    
    Returns:
        test_loss: loss of trained model on testset
        '''

    # Prepare the model for testing
    model.train(mode=False)
    model = model.to(config.device)

    model.eval() # TODO: do I need that
  
    tot_test, tot_acc = 0.0, 0.0
    count = 0

    with torch.no_grad():
        # Count correct predictions for all data points in test set
        for x_batch, y_batch in tester:
            x_batch, y_batch = x_batch.to(config.device), y_batch.to(config.device)

            # Compute average output probabilities over multiple runs
            out = torch.zeros(x_batch.size(0), model.dimensions[-1], device=config.device)
            for s in range(num_samples):
                out += torch.nn.functional.softmax(model(x_batch), dim=1)
            out /= num_samples

            # Batch accuracy
            pred = torch.max(out, dim=1)[1]
        
            acc = pred.eq(y_batch).sum().item()

            tot_acc += acc
            tot_test += x_batch.size()[0]

    return tot_acc / tot_test

