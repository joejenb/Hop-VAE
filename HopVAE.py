import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from hflayers import HopfieldLayer

from utils import get_prior, straight_through_round

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self.conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self.conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=1, padding=2)

        self.conv_4 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self.residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = F.relu(x)
        
        x = self.conv_2(x)
        x = F.relu(x)
        
        x = self.conv_3(x)
        x = F.relu(x)

        x = self.conv_4(x)
        #Should have 2048 units -> embedding_dim * repres_dim^2
        return self.residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self.residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self.conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=1, padding=2)

        self.conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)

        self.conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=out_channels,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv_1(inputs)
        
        x = self.residual_stack(x)
        
        x = self.conv_trans_1(x)
        x = F.relu(x)

        x = self.conv_trans_2(x)
        x = F.relu(x)
        
        return self.conv_trans_3(x)

class HopVAE(nn.Module):
    def __init__(self, config, device):
        super(HopVAE, self).__init__()

        self.device = device

        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embedding_dim
        self.representation_dim = config.representation_dim

        self.encoder = Encoder(config.num_channels, config.num_hiddens,
                                config.num_residual_layers, 
                                config.num_residual_hiddens)

        self.pre_vq_conv = nn.Conv2d(in_channels=config.num_hiddens, 
                                      out_channels=config.embedding_dim,
                                      kernel_size=1, 
                                      stride=1)

        self.hopfield = HopfieldLayer(
                            input_size=config.embedding_dim,                           # R
                            quantity=config.num_embeddings,                             # W_K
                            stored_pattern_as_static=True,
                            state_pattern_as_static=True
                        )

        self.hopfield_joint = HopfieldLayer(
                            input_size=config.embedding_dim,                           # R
                            quantity=config.num_embeddings,                             # W_K
                            stored_pattern_as_static=True,
                            state_pattern_as_static=True
                        )

        self.fit_prior = False
        self.prior = get_prior(config, device)

        self.decoder = Decoder(config.embedding_dim,
                        config.num_channels,
                        config.num_hiddens, 
                        config.num_residual_layers, 
                        config.num_residual_hiddens)

    def sample(self):
        z = self.prior.sample()
        z = z.permute(0, 2, 3, 1).contiguous()

        z_embeddings = z.view(-1, self.representation_dim * self.representation_dim, self.embedding_dim)
        z_embeddings = self.hopfield_joint(z_embeddings)

        z_embeddings = z_embeddings.view(-1, self.representation_dim, self.representation_dim, self.embedding_dim)
        z_embeddings = z_embeddings.permute(0, 3, 1, 2).contiguous()

        x_sample = self.decoder(z_embeddings)

        return x_sample

    def interpolate(self, x, y):
        if (x.size() == y.size()):
            zx = self.encoder(x)
            zx = self.pre_vq_conv(zx)

            zy = self.encoder(y)
            zy = self.pre_vq_conv(zy)

            z = (zx + zy) / 2

            z = z.permute(0, 2, 3, 1).contiguous()
            z = z.view(-1, self.representation_dim * self.representation_dim, self.embedding_dim)

            z_embeddings = self.hopfield(z)
            z = z_embeddings.view(-1, self.representation_dim, self.representation_dim, self.embedding_dim)
            z = z.permute(0, 3, 1, 2).contiguous()
            
            xy_inter = self.decoder(z)

            return xy_inter.detach()

        return x

    def reconstruct(self, x):
        return self.forward(x)

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)

        z = z.permute(0, 2, 3, 1).contiguous()
        z = z.view(-1, self.representation_dim * self.representation_dim, self.embedding_dim)

        z_embeddings = self.hopfield(z)
        z = z_embeddings.view(-1, self.representation_dim, self.representation_dim, self.embedding_dim)
        z = z.permute(0, 3, 1, 2).contiguous()

        if self.fit_prior:

            z_recon, mu, log_var = self.prior(z.detach())
            z_recon = z_recon.permute(0, 2, 3, 1).contiguous()
            z_embeddings_recon = z_recon.view(-1, self.representation_dim * self.representation_dim, self.embedding_dim)
            z_embeddings_recon = self.hopfield_joint(z_embeddings_recon)

            kl_error = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            embedding_recon_loss = F.mse_loss(z_embeddings_recon, z_embeddings)

            x_recon = self.decoder(z)
            return x_recon.detach(), embedding_recon_loss + 0.001 * kl_error


        x_recon = self.decoder(z)

        return x_recon, torch.zeros(1, requires_grad=True).to(self.device)