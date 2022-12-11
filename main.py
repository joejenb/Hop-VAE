import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

import torchvision
from torchvision import transforms

import argparse

import numpy as np
import os

import wandb

from HopVAE import HopVAE

from configs.ffhq_32_config import config as hop_config
from PixelCNN.configs.ffhq_9_config import config as prior_config

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)

args = parser.parse_args()
PATH = args.data 

wandb.init(project="Hop-VAE", config=hop_config)
wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release
class MakeConfig:
    def __init__(self, config):
        self.__dict__ = config

# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config
hop_config = MakeConfig(hop_config)
prior_config = MakeConfig(prior_config)
prior_config.num_channels = hop_config.embedding_dim

def get_data_loaders():
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


def train(model, train_loader, optimiser, scheduler):

    model.train()
    train_res_recon_error = 0

    for X, _ in train_loader:
        X = X.to(model.device)
        optimiser.zero_grad()

        X_recon, Z_prediction_error = model(X)

        recon_error = F.mse_loss(X_recon, X) / config.data_variance
        loss = recon_error + Z_prediction_error
        print(Z_prediction_error)

        loss.backward()
        optimiser.step()
        
        train_res_recon_error += recon_error.item() + Z_prediction_error

    scheduler.step()
    wandb.log({
        "Train Reconstruction Error": (train_res_recon_error) / len(train_loader.dataset)
    })


def test(model, test_loader):
    # Recall Memory
    model.eval() 

    test_res_recon_error = 0

    # Last batch is of different size so simplest to do like this
    iterator = iter(test_loader)
    Y, _ = next(iterator)
    Y = Y.to(model.device)

    Z, _ = next(iterator)
    Z = Z.to(model.device)

    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(model.device)

            X_recon, _ = model(X)
            recon_error = F.mse_loss(X_recon, X) / config.data_variance
            
            test_res_recon_error += recon_error.item()

        ZY_inter = model.interpolate(Z, Y)

        example_images = [wandb.Image(img) for img in X]
        example_reconstructions = [wandb.Image(recon_img) for recon_img in X_recon]
        example_samples = [wandb.Image(model.sample()) for _ in X_recon]
        example_Z = [wandb.Image(recon_img) for recon_img in Z]
        example_Y = [wandb.Image(recon_img) for recon_img in Y]
        example_interpolations = [wandb.Image(inter_img) for inter_img in ZY_inter]

    wandb.log({
        "Test Inputs": example_images,
        "Test Reconstruction": example_reconstructions,
        "Test Interpolations": example_interpolations,
        "Test Samples": example_samples,
        "Test Z": example_Z,
        "Test Y": example_Y,
        "Test Reconstruction Error": test_res_recon_error / len(test_loader.dataset)
        })


def main():

    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, val_loader, test_loader, num_classes = get_data_loaders()
    checkpoint_location = f'checkpoints/{config.data_set}-{config.image_size}.ckpt'
    output_location = f'outputs/{config.data_set}-{config.image_size}.ckpt'

    model = HopVAE(hop_config, prior_config, device).to(device)
    '''if os.path.exists(checkpoint_location):
        #model.load_state_dict(torch.load(checkpoint_location, map_location=device))
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
        model.load_state_dict(pre_state_dict)'''
    model.load_state_dict(torch.load(checkpoint_location, map_location=device))

    optimiser = optim.Adam(model.parameters(), lr=config.learning_rate, amsgrad=False)
    scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=config.gamma)

    wandb.watch(model, log="all")

    for epoch in range(config.epochs):

        if epoch > config.prior_start and not model.fit_prior:
            model.fit_prior = True
            optimiser = optim.Adam(model.prior.parameters(), lr=prior_config.learning_rate)
            scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=prior_config.gamma)

        train(model, train_loader, optimiser, scheduler)
        #test(model, test_loader)

        if not epoch % 5:
            torch.save(model.state_dict(), output_location)

if __name__ == '__main__':
    main()