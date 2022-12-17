import sys
import os
import torch
from torch.utils.data import random_split

import torchvision
from torchvision import transforms


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MakeConfig:
    def __init__(self, config):
        self.__dict__ = config


def straight_through_round(X):
    forward_value = torch.round(X)
    out = X.clone()
    out.data = forward_value.data
    return out


def get_prior(config, device):
    if config.prior == "PixelCNN":
        from Priors.PixelCNN.PixelCNN import PixelCNN as prior
        from Priors.PixelCNN.configs.mnist_8_config import config as prior_config
    elif config.prior == "None":
        prior = lambda x: x
        prior_config = dict()
    
    prior_config = MakeConfig(prior_config)
    prior_config.num_channels = config.index_dim
    prior_config.num_categories = config.num_levels
    return prior(prior_config, device)


def get_data_loaders(config):
    if config.data_set == "MNIST":
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(config.image_size),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        train_set = torchvision.datasets.MNIST(root="/MNIST/", train=True, download=True, transform=transform)
        val_set = torchvision.datasets.MNIST(root="/MNIST/", train=False, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root="/MNIST/", train=False, download=True, transform=transform)
        num_classes = 10
        config.data_variance = 1

    elif config.data_set == "CIFAR10":
        transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(config.image_size),
                transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
            ])
        train_set = torchvision.datasets.CIFAR10(root="/CIFAR10/", train=True, download=True, transform=transform)
        val_set = torchvision.datasets.CIFAR10(root="/CIFAR10/", train=False, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root="/CIFAR10/", train=False, download=True, transform=transform)
        num_classes = 10
        config.data_variance = np.var(train_set.data / 255.0)

    elif config.data_set == "FFHQ":
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(config.image_size),
                transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
            ])

        dataset = torchvision.datasets.ImageFolder(PATH, transform=transform)
        lengths = [int(len(dataset)*0.7), int(len(dataset)*0.1), int(len(dataset)*0.2)]
        train_set, val_set, test_set = random_split(dataset, lengths)

        config.data_variance = 1#np.var(train_set.data / 255.0)
        num_classes = 0

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, num_classes

        



