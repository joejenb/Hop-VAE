import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd.functional import jacobian

import numpy as np

from hflayers import HopfieldLayer

from utils import get_prior, straight_through_round

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            QuantConv2d(in_channels=in_channels,
                        out_channels=num_residual_hiddens,
                        kernel_size=3, stride=1, padding=1, bias=False), 
            nn.ReLU(True),
            QuantConv2d(in_channels=num_residual_hiddens,
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

class QuantConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, num_embeddings=512, quantise=False):
        super(QuantConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_spatial = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv_dim = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.hopfield = HopfieldLayer(
                            input_size=in_channels,                           # R
                            quantity=num_embeddings,                             # W_K
                            stored_pattern_as_static=True,
                            state_pattern_as_static=True
                        )
        self.q_loss = lambda x, y: ((x - y.detach()) ** 2).sum(dim=1)
        self.quantise = quantise

    def q(self, y):
        y_shape = y.shape
        y = y.permute(0, 2, 3, 1).contiguous()
        y = y.view(-1, y_shape[2] * y_shape[3], y_shape[1])

        embeddings = self.hopfield(y)
        embeddings = embeddings.view(-1, y_shape[2], y_shape[3], y_shape[1])

        y_quant = embeddings.permute(0, 3, 1, 2).contiguous()

        return y_quant
    
    def forward(self, x):
        y = self.conv_spatial(x)

        if self.quantise:

            # Quantise and take error
            y_quant = self.q(y)
            y_quant_neg = y_quant.clone()
            y_quant_error = self.q_loss(y, y_quant)

            batch_size = x.shape[0]
            in_repres_dim = x.shape[2]
            out_repres_dim = y.shape[2]

            for batch_num in range(batch_size):
                min_grad, min_vec, min_loc_x, min_loc_y = torch.Tensor([float("inf")]), None, None, None

                for in_loc_x in range(in_repres_dim):
                    for in_loc_y in range(in_repres_dim):
                        in_vec = x[batch_num, :, in_loc_y, in_loc_x]

                        for out_loc_x in range(out_repres_dim):
                            for out_loc_y in range(out_repres_dim):
                                out_vec = y_quant_error[batch_num, out_loc_y, out_loc_x]
                                y_vec = y[batch_num, :, out_loc_y, out_loc_x]

                                #Surely want min -> negative gradient means -> larger vector is i.e 0 or 1 -> lower mse is
                                y_out_grad = torch.autograd.grad(outputs=out_vec, inputs=y, retain_graph=True, create_graph=True)[0]
                                x_y_grad = torch.zeros_like(y_vec)

                                for z_loc in range(self.in_channels):
                                    x_grad = torch.autograd.grad(outputs=y_vec[z_loc], inputs=x, retain_graph=True, create_graph=True)[0]
                                    x_y_grad[z_loc] = x_grad[batch_num, z_loc, in_loc_y, in_loc_x]

                                x_out_grad = x_y_grad * y_out_grad[batch_num, :, in_loc_y, in_loc_x]
                                grad_sum = x_out_grad.sum()

                                if grad_sum < min_grad:
                                    min_grad = grad_sum
                                    min_vec = in_vec * x_y_grad
                                    min_loc_x = out_loc_x
                                    min_loc_y = out_loc_y

                y_quant_neg[batch_num, :, min_loc_y, min_loc_x] -= min_vec
            y_quant_neg = y_quant_neg.view_as(y_quant)
            y_masked = y_quant - y_quant_neg
            return self.conv_dim(self.q(y_masked))

        return self.conv_dim(y)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens//2, kernel_size=1)

        self.conv_2 = QuantConv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4, stride=2, padding=1)

        self.conv_3 = QuantConv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4, stride=2, padding=1)

        self.conv_4 = QuantConv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=3, stride=1, padding=1, num_embeddings=64, quantise=True)

        self.conv_5 = QuantConv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3, stride=1, padding=1)

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
        x = F.relu(x)

        x = self.conv_5(x)
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
                                                kernel_size=3, 
                                                stride=1, padding=1)

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
        self.num_channels = config.num_channels
        self.image_size = config.image_size

        self.encoder = Encoder(config.num_channels, config.num_hiddens,
                                config.num_residual_layers, 
                                config.num_residual_hiddens)

        self.pre_vq_conv = QuantConv2d(in_channels=config.num_hiddens, 
                                      out_channels=config.embedding_dim,
                                      kernel_size=1)

        self.hopfield = HopfieldLayer(
                            input_size=config.embedding_dim,                           # R
                            quantity=config.num_embeddings,                             # W_K
                            stored_pattern_as_static=True,
                            state_pattern_as_static=True
                        )

        self.decoder = Decoder(config.embedding_dim,
                        config.num_channels,
                        config.num_hiddens, 
                        config.num_residual_layers, 
                        config.num_residual_hiddens)

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)

        #64, 64, 8, 8
        z = z.permute(0, 2, 3, 1).contiguous()
        z = z.view(-1, self.representation_dim * self.representation_dim, self.embedding_dim)

        z_embeddings = self.hopfield(z)

        z_embeddings = z_embeddings.view(-1, self.representation_dim, self.representation_dim, self.embedding_dim)
        z_embeddings = z_embeddings.permute(0, 3, 1, 2).contiguous()

        x_recon = self.decoder(z_embeddings)
        return x_recon