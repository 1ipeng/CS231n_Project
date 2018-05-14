import torch
import torch.nn as nn
import numpy as np

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

            {'name':'conv_19', 'out_channels':5, 'kernel_size':1, 'stride':1, 'use_batchnorm': False},
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
        out = torch.sigmoid(out)
        out = out.permute(0, 2, 3, 1)
        return out

    def load_weights(self, weights_dir, num_load_layer):
        own_state = self.state_dict()
        weights = np.load(weights_dir)

        for key in own_state.keys():
            _, layer, name = key.split('.')
            layer_type, index = layer.split('_')
            
            if int(index) >= num_load_layer:
                continue

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
        print('Weights load done.')


def loss_baseline(y_pred, y,l_coord, l_noobj):
    # y (batch_size, num_grid, num_grid, 5)
    # y_pred (batch_size, num_grid, num_grid, 5)
    batch_size, num_grid, _, _ = y.shape

    # Grid cell containing object or not
    obj_mask = (y[:, :, :, 0] == 1)
    noobj_mask = (y[:, :, :, 0] == 0)

    obj_loss_xy = 0
    obj_loss_wh = 0
    obj_loss_pc = 0
    noobj_loss_pc = 0

    # Compute loss for box containing no object
    if len(y_pred[noobj_mask]) != 0:
        noobj_y_pred_pc = y_pred[noobj_mask][:, 0]
        noobj_loss_pc = torch.sum((noobj_y_pred_pc)**2)

    # Compute loss for box containing object
    if len(y_pred[obj_mask]) != 0:
        obj_y_pred_pc = y_pred[obj_mask][:, 0]
        obj_loss_pc = torch.sum((obj_y_pred_pc - 1)**2)

        obj_y_pred_xy = y_pred[obj_mask][:, 1:3]
        obj_y_xy = y[obj_mask][:, 1:3]
        obj_loss_xy = torch.sum((obj_y_pred_xy - obj_y_xy)**2)

        obj_y_pred_wh = y_pred[obj_mask][:, 3:5]
        obj_y_wh = y[obj_mask][:, 3:5]
        obj_loss_wh = torch.sum((torch.sqrt(obj_y_pred_wh) - torch.sqrt(obj_y_wh))**2)

    loss = l_coord * obj_loss_xy + l_coord * obj_loss_wh + obj_loss_pc + l_noobj * noobj_loss_pc
    return loss

# class yolo_v1_loss(nn.Module):
#     def __init__(self, l_coord, l_noobj):
#         self.l_coord = l_coord
#         self.l_noobj = l_noobj

#     def compute_iou(self, box1, box2):
#         '''Compute the intersection over union of two set of boxes
#         Input:
#            box1: (N, 4)
#            box2: (M, 4)
#         Return:
#             iou:(N, M)
#         '''
#         N = box1.shape[0]
#         M = box2.shape[0]

#         lt = torch.max(
#             box1[:,:2].unsqueeze(1).expand(N1,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
#             box2[:,:2].unsqueeze(0).expand(N1,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
#         )

#         rb = torch.min(
#             box1[:,2:].unsqueeze(1).expand(N1,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
#             box2[:,2:].unsqueeze(0).expand(N1,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
#         )

#         wh = rb - lt  # [N,M,2]
#         wh[wh<0] = 0  # clip at 0
#         inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

#         area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
#         area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
#         area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
#         area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

#         iou = inter / (area1 + area2 - inter)
#         return iou

#     def forward(self, y_pred, y):
#         # y (batch_size, num_grid, num_grid, 5)
#         # y_pred (batch_size, num_grid, num_grid, B*5)
#         batch_size, num_grid, _, _ = y.shape
#         B = y_pred.shape[3] / 5

#         # Grid cell containing object or not
#         obj_mask = (y_reshape[:, :, :, :, 0] == 1)
#         noobj_mask = (y_reshape[:, :, :, :, 0] == 0)

#         # Compute loss for grid cells containing no object
#         noobj_y_pred_pc = y_pred[noobj_mask, 0]
#         noobj_y_pc = y[noobj_mask, 0]
#         noobj_loss = torch.sum((noobj_y_pred_pc)**2)

#         # Reshape y, y_pred
#         y_reshape = y.view(batch_size, num_grid, num_grid, B, 5)
#         y_pred_reshape = y_pred.view(batch_size, num_grid, num_grid, 1, 5)
        
# test
'''
model = darknet()
weights_dir = './darknet19_weights.npz'
x = torch.zeros((1, 3, 640, 640))
y = torch.ones((1, 20, 20, 5))
out = model(x)
print(loss_baseline(y, out, 5, 0.5))
# model.load_weights(weights_dir)
'''
