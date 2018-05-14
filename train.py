import numpy as np
import torch
import torch.optim as optim
import os
from tqdm import trange
import model.net as net
from model.data_loader import DataLoader
# from evaluate import evaluate

def train(model, optimizer, loss_fn, data_iterator, metrics, params, num_steps):
    """
    Train the model

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_targets and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and targets
        metrics: (dict) a dictionary of functions that compute a metric using the output and targets of each batch
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()
    
    # Use tqdm for progress bar
    t = trange(num_steps) 
    for i in t:
        x_batch, y_batch = next(data_iterator)
        y_pred_batch = model(x_batch)
        loss = loss_fn(y_pred_batch, y_batch)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        t.set_postfix(loss='{:05.3f}'.format(loss.data[0]))

        # Evaluate summaries only once in a while
        # if i % params.save_summary_steps == 0:
        #     # extract data from torch Variable, move to cpu, convert to numpy arrays
        #     y_pred_batch = y_pred_batch.data.cpu().numpy()
        #     y_batch = y_batch.data.cpu().numpy()

        #     # compute all metrics on this batch
        #     summary_batch = {metric:metrics[metric](y_pred_batch, y_batch)
        #                      for metric in metrics}
        #     summary_batch['loss'] = loss.data[0]
        #     summ.append(summary_batch)

    # compute mean of all metrics in summary
    # metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    # metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    # logging.info("- Train metrics: " + metrics_string)
    

def train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_data: (dict) training data with keys 'data' and 'targets'
        val_data: (dict) validaion data with keys 'data' and 'targets'
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_targets and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and targets of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
        
    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        num_steps = (params.train_size + 1) // params.batch_size
        train_data_iterator = data_loader.data_iterator(train_data, params, shuffle=True)
        train(model, optimizer, loss_fn, train_data_iterator, metrics, params, num_steps)
            
        # Evaluate for one epoch on validation set
        num_steps = (params.val_size + 1) // params.batch_size
        val_data_iterator = data_loader.data_iterator(val_data, params, shuffle=False)
        val_metrics = evaluate(model, loss_fn, val_data_iterator, metrics, params, num_steps)
        
        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()}, 
                               is_best=is_best,
                               checkpoint=model_dir)
            
        # If best_eval, best_save_path        
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc
            
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)