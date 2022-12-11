import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from hflayers import HopfieldLayer
from Residual import ResidualStack

from PixelCNN.PixelCNN import PixelCNN

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

        self.embedding_dim = config.embedding_dim
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
        
        self.fit_prior = False
        self.prior = PixelCNN(prior_config, device)

        self.decoder = Decoder(config.embedding_dim,
                        config.num_channels,
                        config.num_hiddens, 
                        config.num_residual_layers, 
                        config.num_residual_hiddens)

    def sample(self):
        z = self.prior.sample().type(torch.int64)
        z = z.permute(0, 2, 3, 1).contiguous()
        z = z.view(1, self.representation_dim * self.representation_dim, self.embedding_dim)

        z_rounded = torch.round(z * self.num_levels)
        z_rounded_diff = z_rounded - z
        z_rounded = (z + z_rounded_diff) / self.num_levels

        z_quantised = self.hopfield(z_rounded)
        z_quantised = z_quantised.view(-1, self.representation_dim, self.representation_dim, self.embedding_dim)
        z_quantised = z_quantised.permute(0, 3, 1, 2).contiguous()

        z_rounded = torch.round(z_quantised * self.num_levels)
        z_rounded_diff = z_rounded - z_quantised
        z_rounded = z_quantised + z_rounded_diff
        
        x_sample = self._decoder(z_rounded / self.num_levels)

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

            z_rounded = torch.round(z * self.num_levels)
            z_rounded_diff = z_rounded - z
            z_rounded = (z + z_rounded_diff) / self.num_levels

            z_quantised = self.hopfield(z_rounded)
            z_quantised = z_quantised.view(-1, self.representation_dim, self.representation_dim, self.embedding_dim)
            z_quantised = z_quantised.permute(0, 3, 1, 2).contiguous()

            z_rounded = torch.round(z_quantised * self.num_levels)
            z_rounded_diff = z_rounded - z_quantised
            z_rounded = z_quantised + z_rounded_diff

            #start by assuming that num_categories and num_levels are the same 
            z_rounded = self.prior.denoise(z_rounded)

            z_rounded = z_rounded.detach().permute(0, 2, 3, 1).contiguous() / self.num_levels
            z_rounded = z_rounded.view(-1, self.representation_dim * self.representation_dim, self.embedding_dim)

            z_quantised = self.hopfield(z_rounded)
            z_quantised = z_quantised.view(-1, self.representation_dim, self.representation_dim, self.embedding_dim)
            z_quantised = z_quantised.permute(0, 3, 1, 2).contiguous()

            z_pred_rounded = torch.round(z_quantised * self.num_levels)
            z_pred_rounded_diff = z_pred_rounded - z_quantised
            z_pred_rounded = z_quantised + z_pred_rounded_diff
            
            xy_inter = self._decoder(z_pred_rounded / self.num_levels)
            return xy_inter.detach()

        return x

    def forward(self, x):
        '''Need to add in quantisation of vector entries -> say have 512 levels -> multiply by 512
        -> round -> subtract to get difference -> return this value for addition to loss (Don't do this to start with) -> add difference to 
        get rounded value -> Feed through PixelCNN to get classes and loss -> renormalise'''
        '''So hopfield -> project and round -> project back,  for normal case
            hopfield -> project and round -> pixelcnn prior -> project back -> hopfield -> project and round -> project back, for sampling case''' 
        z = self.encoder(x)
        z = F.relu(self.pre_vq_conv(z))

        z = z.permute(0, 2, 3, 1).contiguous()
        z = z.view(-1, self.representation_dim * self.representation_dim, self.embedding_dim)

        z_rounded = torch.round(z * self.num_levels)
        z_rounded_diff = z_rounded - z
        z_rounded = (z + z_rounded_diff) / self.num_levels

        z_quantised = self.hopfield(z_rounded)
        z_quantised = z_quantised.view(-1, self.representation_dim, self.representation_dim, self.embedding_dim)
        z_quantised = z_quantised.permute(0, 3, 1, 2).contiguous()

        z_rounded = torch.round(z_quantised * self.num_levels)
        z_rounded_diff = z_rounded - z_quantised
        z_rounded = z_quantised + z_rounded_diff

        if self.fit_prior:
            #start by assuming that num_categories and num_levels are the same 
            z_pred = self.prior(z_rounded)

            z_cross_entropy = F.cross_entropy(z_pred, z_rounded.long().detach(), reduction='none')
            z_prediction_error = z_cross_entropy.mean(dim=[1,2,3]) * np.log2(np.exp(1))
            z_prediction_error = z_prediction_error.mean()            

            z_pred = z_pred.detach().permute(0, 2, 3, 1).contiguous() / self.num_levels
            z_pred = z_pred.view(-1, self.representation_dim * self.representation_dim, self.embedding_dim)

            z_pred_quantised = self.hopfield(z_pred)
            z_pred_quantised = z_pred_quantised.view(-1, self.representation_dim, self.representation_dim, self.embedding_dim)
            z_pred_quantised = z_pred_quantised.permute(0, 3, 1, 2).contiguous()

            z_pred_rounded = torch.round(z_pred_quantised * self.num_levels)
            z_pred_rounded_diff = z_pred_rounded - z_pred_quantised
            z_pred_rounded = z_pred_quantised + z_pred_rounded_diff
            
            x_recon = self._decoder(z_pred_rounded / self.num_levels)
            return x_recon.detach(), z_prediction_error

        x_recon = self.decoder(z_rounded / self.num_levels)

        return x_recon, 0