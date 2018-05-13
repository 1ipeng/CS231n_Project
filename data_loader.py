import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

def collate_fn(batch):
    # To solve the variable size problem of y
    # Override the default_collate of dataloader
    # Return x:(batch_size, C, H, W) tensor
    #        y:(batch_size,) list, each item 
    x = torch.stack([item[0] for item in batch])
    y = [torch.from_numpy(item[1]) for item in batch]
    return [x, y]

class DetectionDataset(Dataset):
    def __init__(self, x_dir, y_dir, transform):
        self.X = np.load(x_dir)
        self.Y = np.load(y_dir)
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        x = self.transform(x)
        return x, y 

def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    # transform it into a torch tensor
    transformer = transforms.Compose([transforms.ToTensor()]) 

    for split in ['train', 'val', 'test']:
        if split in types:
            x_dir = data_dir + 'X_' + split + '.npy'
            y_dir = data_dir + 'Y_' + split + '.npy'

            dl = DataLoader(DetectionDataset(x_dir, y_dir, transformer), batch_size=params.batch_size, shuffle=True, 
                            num_workers=params.num_workers, pin_memory=params.cuda, collate_fn = collate_fn)
            dataloaders[split] = dl
    return dataloaders
'''
class params(object):
    def __init__(self):
        self.num_workers = 4
        self.batch_size = 32
        self.cuda = False
params = params()

data_dir = '../data/GTSDB/'
test_dataloader = fetch_dataloader(['train'], data_dir, params)
train_loader = test_dataloader['train']
for x, y in train_loader:
    print(x[0], y[0])
    break
'''