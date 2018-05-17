import torch
import torch.nn as nn
import numpy as np

class darknet(nn.Module):
    def __init__(self, params):
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

            {'name':'conv_19', 'out_channels': 5 * params.B, 'kernel_size':1, 'stride':1, 'use_batchnorm': False},
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

    # print('baseline:',noobj_loss_pc, obj_loss_xy, obj_loss_wh, obj_loss_pc)
    return loss

def compute_iou(boxes_pred, boxes_true):
    '''
    Compute intersection over union of two set of boxes
    Args:
      boxes_pred: shape (num_objects, B, 4)
      boxes_true: shape (num_objects, 1, 4)
    Return:
      iou: shape (num_objects, B)
    '''

    num_objects = boxes_pred.size(0)
    B = boxes_pred.size(1)

    lt = torch.max(
        boxes_pred[:,:,:2],                      # [num_objects, B, 2]
        boxes_true[:,:,:2].expand(num_objects, B, 2)    # [num_objects, 1, 2] -> (num_objects, B, 2]
    )

    rb = torch.min(
        boxes_pred[:,:,2:],                      # [num_objects, B, 2]
        boxes_true[:,:,2:].expand(num_objects, B, 2)    # [num_objects, 1, 2] -> (num_objects, B, 2]
    )

    wh = rb - lt # width and height => [num_objects, B, 2]
    wh[wh<0] = 0 # if no intersection, set to zero
    inter = wh[:,:,0] * wh[:,:,1] # [num_objects, B]

    # [num_objects, B, 1] * [num_objects, B, 1] -> [num_objects, B]
    area1 = (boxes_pred[:,:,2]-boxes_pred[:,:,0]) * (boxes_pred[:,:,2]-boxes_pred[:,:,0]) 
    
    # [num_objects, 1, 1] * [num_objects, 1, 1] -> [num_objects, 1] -> [num_objects, B]
    area2 = ((boxes_true[:,:,2]-boxes_true[:,:,0]) * (boxes_true[:,:,2]-boxes_true[:,:,0])).expand(num_objects, B)

    iou = inter / (area1 + area2 - inter) # [num_objects, B]
    
    return iou

def yolo_v1_loss(y_pred, y_true, l_coord, l_noobj):
    # y_pred (batch_size, num_grid, num_grid, B * 5)
    # y_true (batch_size, num_grid, num_grid, 5)
    batch_size, num_grid, _, _ = y_true.shape
    B = y_pred.shape[3] / 5

    # add one dimension to seperate B bounding boxes of y_pred
    y_true = y_true.unsqueeze(-1).view(batch_size, num_grid, num_grid, 1, 5)
    y_pred = y_pred.unsqueeze(-1).view(batch_size, num_grid, num_grid, B, 5) 

    # mask for grid cells with object and wihout object 
    obj_mask = (y_true[:, :, :, 0, 0] == 1) 
    noobj_mask = (y_true[:, :, :, 0, 0] == 0)

    obj_loss_xy = 0
    obj_loss_wh = 0
    obj_loss_pc = 0
    noobj_loss_pc = 0
    
    # Compute loss for boxes in grid cells containing no object
    if len(y_pred[noobj_mask]) != 0:
        noobj_y_pred_pc = y_pred[noobj_mask][:, :, 0]
        noobj_loss_pc = torch.sum((noobj_y_pred_pc)**2)

    # Compute loss for boxes in grid cells containing object
    if len(y_pred[obj_mask]) != 0:
        # boxes coords (xc, yc, w, h) in grid cells with object
        obj_boxes_true = y_true[obj_mask][:, :, 1:5]  #(num_objects, 1, 4)
        obj_boxes_pred = y_pred[obj_mask][:, :, 1:5]  #(num_objects, B, 4)
        obj_pred_pc = y_pred[obj_mask][:, :, 0]  #(num_objects, B)

        # Compute iou between true boxes and B predicted boxes  
        iou = compute_iou(obj_boxes_pred, obj_boxes_true)  #(num_objects, B)

        # Find the target boxes responsible for prediction (boxes with max iou)
        max_iou, max_iou_indices = torch.max(iou, dim=1)

        is_target = torch.zeros(iou.shape)
        is_target[range(iou.shape[0]), max_iou_indices] = 1
        target_mask = (is_target == 1)
        not_target_mask = (is_target == 0)
        # target_mask = (iou == max_iou)
        # not_target_mask = (iou != max_iou)

        # The loss for boxes not responsible for prediction
        not_target_pred_pc = obj_pred_pc[not_target_mask]
        noobj_loss_pc += torch.sum((not_target_pred_pc)**2)

        # The loss for boxes responsible for prediction
        target_pred_pc = obj_pred_pc[target_mask]
        obj_loss_pc = torch.sum((target_pred_pc - 1)**2)

        target_pred_xy = obj_boxes_pred[target_mask][:, 0:2]  #(num_objects, 2)
        target_true_xy = obj_boxes_true[:, 0, 0:2]

        obj_loss_xy = torch.sum((target_pred_xy - target_true_xy)**2)

        target_pred_wh = obj_boxes_pred[target_mask][:, 2:4]
        target_true_wh = obj_boxes_true[:, 0, 2:4]
        obj_loss_wh = torch.sum((torch.sqrt(target_pred_wh) - torch.sqrt(target_true_wh))**2)

    loss = 1./ batch_size * (l_coord * obj_loss_xy + l_coord * obj_loss_wh + obj_loss_pc + l_noobj * noobj_loss_pc)

    # print('v1:',noobj_loss_pc, obj_loss_xy, obj_loss_wh, obj_loss_pc)
    return loss

'''
# Test
model = darknet()
weights_dir = './darknet19_weights.npz'
x = torch.zeros((1, 3, 640, 640))
y = torch.ones((1, 20, 20, 5))
out = model(x)
print(loss_baseline(y, out, 5, 0.5))
# model.load_weights(weights_dir)
'''
