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

class Block1(nn.Module):
    def __init__(self, in_channels, out_channels, num_embeddings=512):
        super(Block1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.hopfield = HopfieldLayer(
                            input_size=out_channels,                           # R
                            quantity=num_embeddings,                             # W_K
                            stored_pattern_as_static=True,
                            state_pattern_as_static=True
                        )
        self.q_loss = lambda x, y: ((x - y.detach()) ** 2).sum(dim=1)

        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels//2,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self.conv_2 = nn.Conv2d(in_channels=out_channels//2,
                                 out_channels=out_channels,
                                 kernel_size=4,
                                 stride=2, padding=1)

    def propagate(self, x):
        y = self.conv_1(x)
        y = F.relu(y)
        
        y = self.conv_2(y)
        y = F.relu(y)
        
        return y

    def q(self, y):
        y_shape = y.shape
        y = y.permute(0, 2, 3, 1).contiguous()
        y = y.view(-1, y_shape[2] * y_shape[3], y_shape[1])

        embeddings = self.hopfield(y)
        embeddings = embeddings.view(-1, y_shape[2], y_shape[3], y_shape[1])

        y_quant = embeddings.permute(0, 3, 1, 2).contiguous()

        return y_quant

    def q_error(self, x):
        y = self.propagate(x)
        y_quant = self.q(y)
        y_quant_error = self.q_loss(y, y_quant)
        return y_quant_error 

    def forward(self, x):
        y = self.propagate(x)

        batch_size = x.shape[0]
        in_repres_dim = x.shape[2]
        out_repres_dim = y.shape[2]

        #batch_num, 1, out_repres_dim, out_repres_dim, batch_size, in_channels, in_repres_dim, in_repres_dim
        e_x_jacob = jacobian(self.q_error, x, create_graph=True)

        #batch_num, in_channels, out_repres_dim, out_repres_dim, batch_size, in_channels, in_repres_dim, in_repres_dim
        x_y_jacob = jacobian(self.propagate, x, create_graph=True)

        x = x.view(batch_size, self.in_channels, -1)
        y_masked = torch.zeros_like(y, requires_grad=True).view(batch_size, self.out_channels, -1)
        y_masked = y_masked.clone()

        for batch_num in range(batch_size):

            #out_repres_dim, out_repres_dim, in_channels, in_repres_dim, in_repres_dim
            e_x_grad = e_x_jacob[batch_num, :, :, batch_num, :, :, :].squeeze()

            #out_repres_dim * out_repres_dim, in_repres_dim * in_repres_dim
            e_x_total = e_x_grad.sum(dim=2).view(out_repres_dim ** 2, in_repres_dim ** 2)

            #in_repres_dim * in_repres_dim
            _, e_x_max_ind = torch.min(e_x_total, dim=0)

            #in_channels, out_repres_dim, out_repres_dim, in_channels, in_repres_dim, in_repres_dim
            x_y_grad = x_y_jacob[batch_num, :, :, :, batch_num, :, :, :].squeeze()

            #in_channels, out_repres_dim * out_repres_dim, in_channels, in_repres_dim * in_repres_dim
            x_y_grad = x_y_grad.view(self.out_channels, out_repres_dim ** 2, self.in_channels, in_repres_dim ** 2)
            
            for ind in range(in_repres_dim ** 2):
                max_ind = e_x_max_ind[ind]
                y_masked[batch_num, :, max_ind] += x[batch_num, :, ind].matmul(x_y_grad[:, max_ind, :, ind].T)

        return self.q(y_masked.view_as(y))

        #return self.q(y)

class Block2(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_layers, num_residual_hiddens, num_embeddings=512):
        super(Block2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.hopfield = HopfieldLayer(
                            input_size=out_channels,                           # R
                            quantity=num_embeddings,                             # W_K
                            stored_pattern_as_static=True,
                            state_pattern_as_static=True
                        )
        self.q_loss = lambda x, y: ((x - y.detach()) ** 2).sum(dim=1)

        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self.residual_stack = ResidualStack(in_channels=in_channels,
                                             num_hiddens=in_channels,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self.conv_2 = nn.Conv2d(in_channels=in_channels, 
                                      out_channels=out_channels,
                                      kernel_size=1, 
                                      stride=1)

    def propagate(self, x):
        y = self.conv_1(x)
        y = self.residual_stack(y)
        y = self.conv_2(y)
        return y

    def q(self, y):
        y_shape = y.shape
        y = y.permute(0, 2, 3, 1).contiguous()
        y = y.view(-1, y_shape[2] * y_shape[3], y_shape[1])

        embeddings = self.hopfield(y)
        embeddings = embeddings.view(-1, y_shape[2], y_shape[3], y_shape[1])

        y_quant = embeddings.permute(0, 3, 1, 2).contiguous()

        return y_quant

    def q_error(self, x):
        y = self.propagate(x)
        y_quant = self.q(y)
        y_quant_error = self.q_loss(y, y_quant)
        return y_quant_error 

    def forward(self, x):
        y = self.propagate(x)

        batch_size = x.shape[0]
        in_repres_dim = x.shape[2]
        out_repres_dim = y.shape[2]

        #batch_num, 1, out_repres_dim, out_repres_dim, batch_size, in_channels, in_repres_dim, in_repres_dim
        e_x_jacob = jacobian(self.q_error, x, create_graph=True)

        #batch_num, in_channels, out_repres_dim, out_repres_dim, batch_size, in_channels, in_repres_dim, in_repres_dim
        x_y_jacob = jacobian(self.propagate, x, create_graph=True)

        x = x.view(batch_size, self.in_channels, -1)
        y_masked = torch.zeros_like(y, requires_grad=True).view(batch_size, self.out_channels, -1)
        y_masked = y_masked.clone()

        for batch_num in range(batch_size):

            #out_repres_dim, out_repres_dim, in_channels, in_repres_dim, in_repres_dim
            e_x_grad = e_x_jacob[batch_num, :, :, batch_num, :, :, :].squeeze()

            #out_repres_dim * out_repres_dim, in_repres_dim * in_repres_dim
            e_x_total = e_x_grad.sum(dim=2).view(out_repres_dim ** 2, in_repres_dim ** 2)

            #in_repres_dim * in_repres_dim
            _, e_x_max_ind = torch.min(e_x_total, dim=0)

            #in_channels, out_repres_dim, out_repres_dim, in_channels, in_repres_dim, in_repres_dim
            x_y_grad = x_y_jacob[batch_num, :, :, :, batch_num, :, :, :].squeeze()

            #in_channels, out_repres_dim * out_repres_dim, in_channels, in_repres_dim * in_repres_dim
            x_y_grad = x_y_grad.view(self.out_channels, out_repres_dim ** 2, self.in_channels, in_repres_dim ** 2)
            
            for ind in range(in_repres_dim ** 2):
                max_ind = e_x_max_ind[ind]
                y_masked[batch_num, :, max_ind] += x[batch_num, :, ind].matmul(x_y_grad[:, max_ind, :, ind].T)

        return self.q(y_masked.view_as(y))

        #return self.q(y)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self.block_1 = Block1(in_channels, out_channels * 2, num_embeddings=64) 

        self.block_2 = Block2(out_channels * 2, out_channels, num_residual_layers, num_residual_hiddens, num_embeddings=768) 

    def forward(self, x):
        y = self.block_1(x)
        y = self.block_2(y)
        return y


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
                                                stride=2, padding=1)

        self.conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=out_channels,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv_1(inputs)
        
        x = self.residual_stack(x)
        
        x = self.conv_trans_1(x)
        x = F.relu(x)

        return self.conv_trans_2(x)


class HopVAE(nn.Module):
    def __init__(self, config, device):
        super(HopVAE, self).__init__()

        self.device = device

        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embedding_dim
        self.representation_dim = config.representation_dim
        self.num_channels = config.num_channels
        self.image_size = config.image_size

        self.encoder = Encoder(config.num_channels, config.embedding_dim,
                                config.num_residual_layers, 
                                config.num_residual_hiddens)

        self.decoder = Decoder(config.embedding_dim,
                        config.num_channels,
                        config.num_hiddens, 
                        config.num_residual_layers, 
                        config.num_residual_hiddens)

    def forward(self, x):
        z = self.encoder(x)

        x_recon = self.decoder(z)

        return x_recon