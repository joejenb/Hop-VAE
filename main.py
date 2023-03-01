import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms

import argparse

import numpy as np
import os

import wandb

from HopVAE import HopVAE

from utils import get_data_loaders, load_from_checkpoint, MakeConfig, save_checkpoint

from configs.OSTerrain50_64_config import config

wandb.init(project="Hop-VAE", config=config)
config = MakeConfig(config)

def train(model, train_loader, optimiser, scheduler):

    model.train()
    train_res_recon_error = 0

    for X, _ in train_loader:
        X = X.to(model.device)

        optimiser.zero_grad()

        X_neg_log_probs = model(X)

        recon_error = torch.mean(X_neg_log_probs)
        loss = recon_error

        loss.backward()
        optimiser.step()
        
        train_res_recon_error += recon_error.item()

    scheduler.step()
    wandb.log({
        "Train Reconstruction Error": (train_res_recon_error) / len(train_loader.dataset),
    })


def test(model, test_loader):
    model.eval() 

    test_res_recon_error = 0

    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(model.device)

            X_recon = model(X)
            recon_error = F.mse_loss(X_recon, X)
            
            test_res_recon_error += recon_error.item()

        example_images = [wandb.Image(img) for img in X]
        example_reconstructions = [wandb.Image(recon_img) for recon_img in X_recon]

    wandb.log({
        "Test Inputs": example_images,
        "Test Reconstruction": example_reconstructions,
        "Test Reconstruction Error": test_res_recon_error / len(test_loader.dataset)
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)

    args = parser.parse_args()
    PATH = args.data 

    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, val_loader, test_loader = get_data_loaders(config, PATH)
    checkpoint_location = f'checkpoints/{config.data_set}-{config.image_size}.ckpt'
    output_location = f'outputs/{config.data_set}-{config.image_size}.ckpt'

    model = HopVAE(config, device).to(device)

    optimiser = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=config.gamma)

    model, optimiser, epoch = load_from_checkpoint(model, optimiser, checkpoint_location)

    wandb.watch(model, log="all")

    for epoch in range(epoch, config.epochs):

        train(model, train_loader, optimiser, scheduler)

        if not epoch % 5:
            test(model, test_loader)

        if not epoch % 5:
            save_checkpoint(model, optimiser, epoch, output_location)

if __name__ == '__main__':
    main()