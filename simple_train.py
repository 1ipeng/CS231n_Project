from model.net import *
import torch.nn as nn
import torch.optim as optim
import torch
from model.data_loader import fetch_dataloader
import utils
import matplotlib.pyplot as plt

print("Start training")

params = utils.Params('./experiment/params.json')
# params.cuda = torch.cuda.is_available()
if params.cuda:
    device = torch.device('cuda')
    print('device:cuda')
else:
    device = torch.device('cpu')
    print('device:cpu')

data_dir = './data/GTSDB/'
data_loader = fetch_dataloader(['train', 'val', 'test'], data_dir, params)
loader_train = data_loader['train']
loader_val = data_loader['val']
weights_dir = './darknet19_weights.npz'
dtype = torch.float32

train_losses = []
val_losses = []

def check_val_loss(loader, model, loss_fn): 
    loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)
            y_pred = model(x)
            minibatch_loss = loss_fn(y_pred, y, params.l_coord, params.l_noobj)
            loss += minibatch_loss
    return loss


def simple_train(model, optimizer, loss_fn, epochs=10):
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)
            y_pred = model(x)

            loss = loss_fn(y_pred, y, params.l_coord, params.l_noobj)
            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()
            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()
            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            print(t, 'loss:', loss)

    return model

def simple_predict(model):
    for x,y in loader_train:
        y_pred = model(x)

        y_pred = y_pred.data.numpy()
        x = x.data.numpy().transpose(0,2,3,1)
        y = y.data.numpy()
        
        np.save('./data/tiny_GTSDB/x_pred', x)
        np.save('./data/tiny_GTSDB/Y_pred', y_pred)
        
        plt.figure()
        utils.draw_boxes_output(x[0], y_pred[0])
        plt.figure()
        utils.draw_boxes_output(x[0], y[0])
        plt.show()
        break

model = darknet(params).cuda() if params.cuda else darknet(params)
# model.load_weights(weights_dir, 18)
optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
simple_train(model, optimizer, yolo_v1_loss, params.num_epochs)
# simple_predict(model)