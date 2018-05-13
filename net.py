import torch
import torch.nn as nn
import numpy as np
from util import *

class darknet(nn.Module):
    def __init__(self):
        super().__init__()
        net_cfg = [
            {'name':'conv_1', 'out_channels':32, 'kernel_size':3, 'stride':1, 'use_batchnorm': True},
            {'name':'maxpool_1', 'kernel_size':2, 'stride':2},
           
            {'name':'conv_2', 'out_channels':64, 'kernel_size':3, 'stride':1, 'use_batchnorm': True},
            {'name':'maxpool_2', 'kernel_size':2, 'stride':2},
            
            {'name':'conv_3', 'out_channels':128, 'kernel_size':3, 'stride':1, 'use_batchnorm': True},
            {'name':'conv_4', 'out_channels':64, 'kernel_size':1, 'stride':1, 'use_batchnorm': True},
            {'name':'conv_5', 'out_channels':128, 'kernel_size':3, 'stride':1, 'use_batchnorm': True},
            {'name':'maxpool_3', 'kernel_size':2, 'stride':2},
            
            {'name':'conv_6', 'out_channels':256, 'kernel_size':3, 'stride':1, 'use_batchnorm': True},
            {'name':'conv_7', 'out_channels':128, 'kernel_size':1, 'stride':1, 'use_batchnorm': True},
            {'name':'conv_8', 'out_channels':256, 'kernel_size':3, 'stride':1, 'use_batchnorm': True},
            {'name':'maxpool_4', 'kernel_size':2, 'stride':2},

            {'name':'conv_9', 'out_channels':512, 'kernel_size':3, 'stride':1, 'use_batchnorm': True},
            {'name':'conv_10', 'out_channels':256, 'kernel_size':1, 'stride':1, 'use_batchnorm': True},
            {'name':'conv_11', 'out_channels':512, 'kernel_size':3, 'stride':1, 'use_batchnorm': True},
            {'name':'conv_12', 'out_channels':256, 'kernel_size':1, 'stride':1, 'use_batchnorm': True},
            {'name':'conv_13', 'out_channels':512, 'kernel_size':3, 'stride':1, 'use_batchnorm': True},
            {'name':'maxpool_5', 'kernel_size':2, 'stride':2},

            {'name':'conv_14', 'out_channels':1024, 'kernel_size':3, 'stride':1, 'use_batchnorm': True},
            {'name':'conv_15', 'out_channels':512, 'kernel_size':1, 'stride':1, 'use_batchnorm': True},
            {'name':'conv_16', 'out_channels':1024, 'kernel_size':3, 'stride':1, 'use_batchnorm': True},
            {'name':'conv_17', 'out_channels':512, 'kernel_size':1, 'stride':1, 'use_batchnorm': True},
            {'name':'conv_18', 'out_channels':1024, 'kernel_size':3, 'stride':1, 'use_batchnorm': True},

            {'name':'conv_19', 'out_channels':1000, 'kernel_size':1, 'stride':1, 'use_batchnorm': False},
        ]

        self.conv = torch.nn.Sequential()

        out_channels = 3
        for layer_cfg in net_cfg:
            out_channels = self.make_layer(layer_cfg, out_channels)

    def make_layer(self, layer_cfg, in_channels):
        name = layer_cfg['name']
        layer_type, index = name.split('_')

        if layer_type == 'conv':
            out_channels = layer_cfg['out_channels']
            padding = layer_cfg['kernel_size'] // 2
            self.conv.add_module(name, nn.Conv2d(in_channels, out_channels, layer_cfg['kernel_size'], stride=layer_cfg['stride'], padding=padding, bias=False))
            if layer_cfg['use_batchnorm']:
                self.conv.add_module('bn_' + index, nn.BatchNorm2d(out_channels, momentum=0.01))
            self.conv.add_module('relu_' + index, nn.LeakyReLU(0.1))

        if layer_type == 'maxpool':
            out_channels = in_channels
            self.conv.add_module(name, nn.MaxPool2d(layer_cfg['kernel_size'], stride=layer_cfg['stride']))

        if layer_type == 'fc': 
            out_channels = layer_cfg['out_channels']
            self.conv.add_module(name, nn.Linear(in_channels, out_channels, bias=True))

        return out_channels
    
    def forward(self, x):
        # forward always defines connectivity
        out = self.conv.forward(x)
        return out

    def load_weights(self, weights_dir):
        own_state = self.state_dict()
        weights = np.load(weights_dir)

        for key in own_state.keys():
            _, layer, name = key.split('.')
            layer_type, index = layer.split('_')
            param_name = str(int(index) - 1) + '-convolutional/' 
            if layer_type == 'conv' and name == 'weight':
                param_name += 'kernel:0'
            if layer_type == 'bn' and name == 'weight':
                param_name += 'gamma:0'
            if name == 'bias':
                param_name += 'biases:0'
            if name == 'running_mean':
                param_name += 'moving_mean:0'
            if name == 'running_var':
                param_name += 'moving_variance:0'
            
            param = torch.from_numpy(weights[param_name])
            
            if layer_type == 'conv' and name == 'weight':
                param = param.permute(3, 2, 0, 1)
            own_state[key].copy_(param)

def loss(self, y, y_pred, params):
    # y (batch_size,)  each entry(num_box, 4)
    # y_pred (batch_size, 13, 13, 10)
    for i in params.batch_size:
        mask = np.zeros((params.num_grid, params.num_grid, params.num_anchor))
        boxes_xy = y[i]
        for box_xy in boxes_xy:
            box_cwh = xy_to_cwh(box_xy)
            normalized_cwh, positon = normalize_box_cwh(params.image_resize, param.num_grid, box_cwh)
            
            mask[positon, ] = 1


    loss_noobj = lambda_noobj * torch.sum(np.y_pred[:, :, :, 0::5]**2)









'''
# test
model = darknet()
weights_dir = './darknet19_weights.npz'
# x = torch.zeros((64, 3, 416, 416))
# out = model.forward(x)
# print(out.shape)
model.load_weights(weights_dir)
'''
