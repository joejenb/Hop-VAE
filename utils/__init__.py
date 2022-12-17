import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

import torchvision
from torchvision import transforms


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MakeConfig:
    def __init__(self, config):
        self.__dict__ = config

class Normal(nn.Module):
    def __init__(self, config, device):
        self.device = device
        self.config = config
        self.dummy_module = nn.Linear(1, 1)

    def sample(self):
        return torch.randn(1, self.config.embedding_dim, self.config.representation_dim, self.config.representation_dim).to(self.device)
    
    def interpolate(self, X, Y):
        return (X + Y) / 2
    
    def reconstruct(self, X):
        return X

    def forward(self, X):
        return X

def load_from_checkpoint(model, checkpoint_location):
    if os.path.exists(checkpoint_location):
        pre_state_dict = torch.load(checkpoint_location, map_location=device)
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

def get_prior_optimiser(config, prior):

    if config.prior == "PixelCNN":
        from priors.PixelCNN.configs.mnist_8_config import config as prior_config

    prior_config = MakeConfig(prior_config)
    optimiser = optim.Adam(prior.parameters(), lr=prior_config.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=prior_config.gamma)

    return optimiser, scheduler

def get_prior(config, device):
    if config.prior == "PixelCNN":
        from priors.PixelCNN.PixelCNN import PixelCNN as prior
        from priors.PixelCNN.configs.mnist_8_config import config as prior_config
    elif config.prior == "None":
        prior = Normal
        prior_config = dict()
    
    prior_config = MakeConfig(prior_config)
    prior_config.num_channels = config.index_dim
    prior_config.num_categories = config.num_levels
    return prior(prior_config, device)


def get_data_loaders(config, PATH):
    if config.data_set == "MNIST":
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(config.image_size),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        train_set = torchvision.datasets.MNIST(root=PATH, train=True, download=True, transform=transform)
        val_set = torchvision.datasets.MNIST(root=PATH, train=False, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=PATH, train=False, download=True, transform=transform)
        num_classes = 10
        config.data_variance = 1

    elif config.data_set == "CIFAR10":
        transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(config.image_size),
                transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
            ])
        train_set = torchvision.datasets.CIFAR10(root=PATH, train=True, download=True, transform=transform)
        val_set = torchvision.datasets.CIFAR10(root=PATH, train=False, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=PATH, train=False, download=True, transform=transform)
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

        



