import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

import torchvision
from torchvision import transforms

from datasets.OSTerrain50 import OSTerrain50

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MakeConfig:
    def __init__(self, config):
        self.__dict__ = config

def load_from_checkpoint(model, checkpoint_location):
    if os.path.exists(checkpoint_location):
        pre_state_dict = torch.load(checkpoint_location, map_location=model.device)
        to_delete = []
        for key in pre_state_dict.keys():
            if key not in model.state_dict().keys():
                to_delete.append(key)
        for key in to_delete:
            del pre_state_dict[key]
        for key in model.state_dict().keys():
            if key not in pre_state_dict.keys():
                pre_state_dict[key] = model.state_dict()[key]
        model.load_state_dict(pre_state_dict)
    return model

def straight_through_round(X):
    forward_value = torch.round(X)
    out = X.clone()
    out.data = forward_value.data
    return out

def get_data_loaders(config, PATH):
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(config.image_size),
            transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
        ])

    if config.data_set == "MNIST":
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(config.image_size),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        train_set = torchvision.datasets.MNIST(root=PATH, train=True, download=True, transform=transform)
        val_set = torchvision.datasets.MNIST(root=PATH, train=False, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=PATH, train=False, download=True, transform=transform)

    elif config.data_set == "CIFAR10":
        train_set = torchvision.datasets.CIFAR10(root=PATH, train=True, download=True, transform=transform)
        val_set = torchvision.datasets.CIFAR10(root=PATH, train=False, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=PATH, train=False, download=True, transform=transform)

    elif config.data_set == "FFHQ":
        dataset = torchvision.datasets.ImageFolder(PATH, transform=transform)
        lengths = [int(len(dataset)*0.7), int(len(dataset)*0.1), int(len(dataset)*0.2)]
        train_set, val_set, test_set = random_split(dataset, lengths)
    
    elif config.data_set == "CELEBA":
        train_set = torchvision.datasets.CelebA(root=PATH, split='train', download=False, transform=transform, target_type="identity")
        val_set = torchvision.datasets.CelebA(root=PATH, split='valid', download=False, transform=transform, target_type="identity")
        test_set = torchvision.datasets.CelebA(root=PATH, split='test', download=False, transform=transform, target_type="identity")
    
    elif config.data_set == "OSTerrain50":
        train_set = OSTerrain50(root=PATH, split='train', transform=transforms.RandomCrop(config.image_size))
        val_set = OSTerrain50(root=PATH, split='val', transform=transforms.RandomCrop(config.image_size))
        test_set = OSTerrain50(root=PATH, split='test', transform=transforms.RandomCrop(config.image_size)) 
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

        



