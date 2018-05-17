import numpy as np
import torch
import os
from tqdm import trange
import logging
import utils

def train(model, optimizer, loss_fn, dataloader, params):
    """
    Train the model for one epoch
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_targets and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and targets
        params: (Params) hyperparameters
    """

    # set model to training mode
    avg_loss = 0
    num_steps = len(dataloader)
    # Use tqdm for progress bar
    t = trange(num_steps) 

    for i, (x_batch, y_batch) in enumerate(dataloader):
        model.train()
        x_batch = x_batch.to(device=params.device, dtype=torch.float32)
        y_batch = y_batch.to(device=params.device, dtype=torch.float32)
        
        y_pred_batch = model(x_batch)
        loss = loss_fn(y_pred_batch, y_batch, params)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        t.set_postfix(loss='{:05.3f}'.format(loss.item())) 
        t.update()

        avg_loss += loss / num_steps
        
    return avg_loss

def evaluate(model, loss_fn, dataloader, params):
    """Evaluate the model

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        params: (Params) hyperparameters
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    avg_loss = 0
    num_steps = len(dataloader)

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device=params.device, dtype=torch.float32)
            y_batch = y_batch.to(device=params.device, dtype=torch.float32)
            
            y_pred_batch = model(x_batch)
            loss = loss_fn(y_pred_batch, y_batch, params)
            avg_loss += loss / num_steps

    return avg_loss

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_data: (dict) training data with keys 'data' and 'targets'
        val_data: (dict) validaion data with keys 'data' and 'targets'
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_targets and computes the loss for the batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        print("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
    
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')

    for epoch in range(params.num_epochs):
        # Run one epoch
        print("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_loss = train(model, optimizer, loss_fn, train_dataloader, params)
            
        # Evaluate for one epoch on validation set
        val_loss = evaluate(model, loss_fn, val_dataloader, params)

        train_loss = train_loss.item()
        val_loss = val_loss.item()

        is_best = val_loss < best_val_loss

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()}, 
                               is_best=is_best,
                               checkpoint=model_dir)
            
        # If best_eval, best_save_path        
        if is_best:
            best_val_loss = val_loss

        print('Train loss:' + '{:05.3f}'.format(train_loss), ', Val loss:' + '{:05.3f}'.format(val_loss), 'Best val loss:', best_val_loss)
            
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        np.save(os.path.join(model_dir, 'train_losses'), train_losses)
        np.save(os.path.join(model_dir, 'val_losses'), val_losses)