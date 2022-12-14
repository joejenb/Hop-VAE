import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from hflayers import HopfieldLayer
from Residual import ResidualStack

from PixelCNN.PixelCNN import PixelCNN


def straight_through_round(X):
    forward_value = torch.round(X)
    out = X.clone()
    out.data = forward_value.data
    return out

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
    def __init__(self, config, prior_config, device):
        super(HopVAE, self).__init__()

        self.device = device

        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embedding_dim
        self.index_dim = config.index_dim
        self.representation_dim = config.representation_dim
        self.num_levels = config.num_levels

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

        self.embedding_to_index = HopfieldLayer(
                            input_size=config.embedding_dim,                           # R
                            output_size=config.index_dim,
                            quantity=config.num_embeddings,                             # W_K
                            stored_pattern_as_static=True,
                            state_pattern_as_static=True
                        )
        
        self.index_to_embedding = HopfieldLayer(
                            input_size=config.index_dim,                           # R
                            output_size=config.embedding_dim,
                            quantity=config.num_embeddings,                             # W_K
                            stored_pattern_as_static=True,
                            state_pattern_as_static=True
                        )


        self.post_vq_conv = nn.Conv2d(in_channels=config.index_dim, 
                                      out_channels=config.index_dim,
                                      kernel_size=1, 
                                      stride=1)
        
        self.fit_prior = False
        self.prior = PixelCNN(prior_config, device)

        self.decoder = Decoder(config.embedding_dim,
                        config.num_channels,
                        config.num_hiddens, 
                        config.num_residual_layers, 
                        config.num_residual_hiddens)

    def sample(self):
        z_indices = self.prior.sample().type(torch.int64) / (self.num_levels - 1)

        z_indices = z_indices.permute(0, 2, 3, 1).contiguous()
        z_indices = z_indices.view(-1, self.representation_dim * self.representation_dim, self.index_dim)

        z_embeddings = self.index_to_embedding(z_indices)

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
            z_indices = self.embedding_to_index(z_embeddings)

            #z_indices = z_indices.view(-1, self.representation_dim, self.representation_dim, self.index_dim)
            #z_indices = z_indices.permute(0, 3, 1, 2).contiguous()

            z_indices = F.relu(z_indices)#self.post_vq_conv(z_indices))
            z_indices = 1 - F.relu(1 - z_indices)
            #z_indices = F.sigmoid(self.post_vq_conv(z_indices))
            #z_indices_quantised = straight_through_round(z_indices * (self.num_levels - 1))

            z_indices_quantised = z_indices_quantised.view(-1, self.representation_dim, self.representation_dim, self.index_dim)
            z_indices_quantised = z_indices_quantised.permute(0, 3, 1, 2).contiguous()

            z_indices = self.prior.denoise(z_indices_quantised) / (self.num_levels - 1)

            z_indices = z_indices.permute(0, 2, 3, 1).contiguous()
            z_indices = z_indices.view(-1, self.representation_dim * self.representation_dim, self.index_dim)

            z_embeddings = self.index_to_embedding(z_indices)

            z_embeddings = z_embeddings.view(-1, self.representation_dim, self.representation_dim, self.embedding_dim)
            z_embeddings = z_embeddings.permute(0, 3, 1, 2).contiguous()

            xy_inter = self.decoder(z_embeddings)

            return xy_inter.detach()

        return x

    def forward(self, x):
        z = self.encoder(x)
        z = F.relu(self.pre_vq_conv(z))

        z = z.permute(0, 2, 3, 1).contiguous()
        z = z.view(-1, self.representation_dim * self.representation_dim, self.embedding_dim)

        z_embeddings = self.hopfield(z)

        z_indices = self.embedding_to_index(z_embeddings)

        #z_indices = z_indices.view(-1, self.representation_dim, self.representation_dim, self.index_dim)
        #z_indices = z_indices.permute(0, 3, 1, 2).contiguous()

        z_indices = F.relu(z_indices)#self.post_vq_conv(z_indices))
        z_indices = 1 - F.relu(1 - z_indices)
        #z_indices = F.sigmoid(self.post_vq_conv(z_indices))
        #z_indices_quantised = straight_through_round(z_indices * (self.num_levels - 1))
        z_indices = z_indices_quantised / (self.num_levels - 1)

        z_indices = z_indices.permute(0, 2, 3, 1).contiguous()
        z_indices = z_indices.view(-1, self.representation_dim * self.representation_dim, self.index_dim)
        z_embeddings = self.index_to_embedding(z_indices)

        z_embeddings = z_embeddings.view(-1, self.representation_dim, self.representation_dim, self.embedding_dim)
        z_embeddings = z_embeddings.permute(0, 3, 1, 2).contiguous()

        if self.fit_prior:
            #start by assuming that num_categories and num_levels are the same 
            z_indices_quantised = z_indices_quantised.view(-1, self.representation_dim, self.representation_dim, self.index_dim)
            z_indices_quantised = z_indices_quantised.permute(0, 3, 1, 2).contiguous()

            z_pred = self.prior(z_indices_quantised.detach())

            z_cross_entropy = F.cross_entropy(z_pred, z_indices_quantised.long().detach(), reduction='none')
            z_prediction_error = z_cross_entropy.mean(dim=[1,2,3]) * np.log2(np.exp(1))
            z_prediction_error = z_prediction_error.mean()            

            x_recon = self.decoder(z_embeddings)
            return x_recon.detach(), z_prediction_error


        x_recon = self.decoder(z_embeddings)

        return x_recon, torch.zeros(1, requires_grad=True).to(self.device)