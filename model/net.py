import torch
import torch.nn as nn

class darknet(nn.Module):
    def __init__(self):
        super().__init__()
        net_cfg = [
            {'type':'conv', 'out_channels':32, 'kernel_size':3, 'stride':1},
            {'type':'maxpool', 'kernel_size':2, 'stride':2},
           
            {'type':'conv', 'out_channels':64, 'kernel_size':3, 'stride':1},
            {'type':'maxpool', 'kernel_size':2, 'stride':2},
            
            {'type':'conv', 'out_channels':128, 'kernel_size':3, 'stride':1},
            {'type':'conv', 'out_channels':64, 'kernel_size':1, 'stride':1},
            {'type':'conv', 'out_channels':128, 'kernel_size':3, 'stride':1},
            {'type':'maxpool', 'kernel_size':2, 'stride':2},
            
            {'type':'conv', 'out_channels':256, 'kernel_size':3, 'stride':1},
            {'type':'conv', 'out_channels':128, 'kernel_size':1, 'stride':1},
            {'type':'conv', 'out_channels':256, 'kernel_size':3, 'stride':1},
            {'type':'maxpool', 'kernel_size':2, 'stride':2},

            {'type':'conv', 'out_channels':512, 'kernel_size':3, 'stride':1},
            {'type':'conv', 'out_channels':256, 'kernel_size':1, 'stride':1},
            {'type':'conv', 'out_channels':512, 'kernel_size':3, 'stride':1},
            {'type':'conv', 'out_channels':256, 'kernel_size':1, 'stride':1},
            {'type':'conv', 'out_channels':512, 'kernel_size':3, 'stride':1},
            {'type':'maxpool', 'kernel_size':2, 'stride':2},

            {'type':'conv', 'out_channels':1024, 'kernel_size':3, 'stride':1},
            {'type':'conv', 'out_channels':512, 'kernel_size':1, 'stride':1},
            {'type':'conv', 'out_channels':1024, 'kernel_size':3, 'stride':1},
            {'type':'conv', 'out_channels':512, 'kernel_size':1, 'stride':1},
            {'type':'conv', 'out_channels':1024, 'kernel_size':3, 'stride':1},

            {'type':'conv', 'out_channels':1000, 'kernel_size':1, 'stride':1},
        ]

        self.layers = []
        out_channels = 3
        for layer_cfg in net_cfg:
            layer, out_channels = self.make_layer(layer_cfg, out_channels)
            self.layers.append(layer)

    def make_layer(self, layer_cfg, in_channels):
        if layer_cfg['type'] == 'conv':
            out_channels = layer_cfg['out_channels']
            padding = layer_cfg['kernel_size'] // 2
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, layer_cfg['kernel_size'], stride=layer_cfg['stride'], padding=padding, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1)
            )

        if layer_cfg['type'] == 'maxpool':
            out_channels = in_channels
            layer = nn.MaxPool2d(layer_cfg['kernel_size'], stride=layer_cfg['stride'])

        if layer_cfg['type'] == 'fc':
            out_channels = layer_cfg['out_channels']
            layer = nn.Linear(in_channels, out_channels, bias=True)

        return layer, out_channels
    
    def forward(self, x):
        # forward always defines connectivity
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            print(i, out.shape)
        return out

    def loss(self, y, y_pred):
        pass

model = darknet()
x = torch.zeros((64, 3, 448, 448))
out = model.forward(x)
