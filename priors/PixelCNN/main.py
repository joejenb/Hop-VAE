import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

import torchvision
from torchvision import transforms

import argparse
import dill

import numpy as np
import os

import wandb

from PixelCNN import PixelCNN
from configs.mnist_28_config import config

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)

args = parser.parse_args()
PATH = args.data 

wandb.init(project="PixelCNN", config=config)
wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config

def discretize(sample):
    return (sample * 255).to(torch.long)

def get_data_loaders():
    if config.data_set == "MNIST":
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(config.image_size),
                discretize
                #transforms.Normalize((0.1307,), (0.3081,))
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

        X_logits = model(X)
        cross_entropy = F.cross_entropy(X_logits, X, reduction='none')
        prediction_error = cross_entropy.mean(dim=[1,2,3]) * np.log2(np.exp(1))
        loss = prediction_error.mean()

        loss.backward()
        optimiser.step()
        
        train_res_recon_error += loss.item()

    scheduler.step()
    wandb.log({
        "Train Reconstruction Error": train_res_recon_error / len(train_loader.dataset)
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

            X_logits = model(X)
            cross_entropy = F.cross_entropy(X_logits, X, reduction='none')
            prediction_error = cross_entropy.mean(dim=[1,2,3]) * np.log2(np.exp(1))
            loss = prediction_error.mean()
   
            test_res_recon_error += loss.item()

        ZY_inter = model.interpolate(Z, Y)
        X_sample = model.sample()

        example_images = [wandb.Image(img.float() / 255.0) for img in X]
        example_sample = [wandb.Image(recon_sample.float() / 255.0) for recon_sample in X_sample]
        example_Z = [wandb.Image(recon_img.float() / 255.0) for recon_img in Z]
        example_Y = [wandb.Image(recon_img.float() / 255.0) for recon_img in Y]
        example_interpolations = [wandb.Image(inter_img.float() / 255.0) for inter_img in ZY_inter]

    wandb.log({
        "Test Inputs": example_images,
        "Test Sample": example_sample,
        "Test Interpolations": example_interpolations,
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

    ### Add in correct parameters
    #model = PixelCNN(config, device).to(device)
    model = PixelCNN(config, device).to(device)

    if os.path.exists(checkpoint_location):
        model.load_state_dict(torch.load(checkpoint_location, map_location=device))

    optimiser = optim.Adam(model.parameters(), lr=config.learning_rate, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimiser, 1, gamma=config.gamma)

    wandb.watch(model, log="all")

    for epoch in range(config.epochs):

        train(model, train_loader, optimiser, scheduler)
        test(model, test_loader)

        if not epoch % 5:
            torch.save(model.state_dict(), output_location)

if __name__ == '__main__':
    main()