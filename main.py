import torch
import torch.nn.functional as F
import torch.optim as optim

import argparse

import numpy as np
import os

import wandb

from HopVAE import HopVAE

from utils import get_data_loaders, get_prior_optimiser, load_from_checkpoint, MakeConfig

from configs.cifar10_32_config import config

wandb.init(project="Hop-VAE", config=config)
config = MakeConfig(config)

def train(model, train_loader, optimiser, scheduler):

    model.train()
    train_res_recon_error = 0

    iter_num = 0
    for X, _ in train_loader:
        print(iter_num)
        if iter_num > 50:
            break
        
        iter_num += 1
        X = X.to(model.device)
        optimiser.zero_grad()

        X_recon = model(X)

        recon_error = F.mse_loss(X_recon, X)
        loss = recon_error

        loss.backward()
        optimiser.step()
        
        train_res_recon_error += recon_error.item()

    scheduler.step()
    wandb.log({
        "Train Reconstruction Error": (train_res_recon_error) / len(train_loader)
    })


def test(model, test_loader):
    # Recall Memory
    model.eval() 

    test_res_recon_error = 0

    #with torch.no_grad():
    iter_num = 0
    for X, _ in test_loader:
        print(iter_num)
        if iter_num > 0:
            break

        iter_num += 1
        X = X.to(model.device)

        X_recon = model(X)
        recon_error = F.mse_loss(X_recon, X)
        
        test_res_recon_error += recon_error.item()

    example_images = [wandb.Image(img) for img in X]
    example_reconstructions = [wandb.Image(recon_img) for recon_img in X_recon]

    wandb.log({
        "Test Inputs": example_images,
        "Test Reconstruction": example_reconstructions,
        "Test Reconstruction Error": test_res_recon_error / len(test_loader)
        })

num_embeddings = [32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 3072, 4096, 5120]
num_embeddings.reverse()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)

    args = parser.parse_args()
    PATH = args.data 

    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, val_loader, test_loader, num_classes = get_data_loaders(config, PATH)
    checkpoint_location = f'checkpoints/{config.data_set}-{config.image_size}.ckpt'
    output_location = f'outputs/{config.data_set}-{config.image_size}.ckpt'

    model = HopVAE(config, device).to(device)
    model = load_from_checkpoint(model, checkpoint_location)

    optimiser = optim.Adam(model.parameters(), lr=config.learning_rate, amsgrad=False)
    scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=config.gamma)

    wandb.watch(model, log="all")

    for epoch in range(config.epochs):

        train(model, train_loader, optimiser, scheduler)

        if not epoch % 5:
            test(model, test_loader)

        if not epoch % 5:
            torch.save(model.state_dict(), output_location)

if __name__ == '__main__':
    main()