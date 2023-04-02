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
        self.in_repres_dim = 32
        self.out_repres_dim = 8

        self.hopfield = HopfieldLayer(
                            input_size=out_channels,                           # R
                            quantity=num_embeddings,                             # W_K
                            stored_pattern_as_static=True,
                            state_pattern_as_static=True
                        )
                        
        self.q_loss = lambda x, y: ((x - y.detach()) ** 2)

        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels//2,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self.conv_2 = nn.Conv2d(in_channels=out_channels//2,
                                 out_channels=out_channels,
                                 kernel_size=4,
                                 stride=2, padding=1)
        
        self.e_q = nn.ConvTranspose2d(in_channels=out_channels, 
                                                out_channels=self.out_repres_dim * self.out_repres_dim * out_channels,
                                                kernel_size=3, 
                                                stride=1, padding=1)

        self.q_z_2 = nn.ConvTranspose2d(in_channels=out_channels, 
                                                out_channels=out_channels,
                                                kernel_size=3, 
                                                stride=1, padding=1)

        self.q_z_1 = nn.ConvTranspose2d(in_channels=out_channels, 
                                                out_channels=out_channels//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        #
        self.q_x = nn.ConvTranspose2d(in_channels=out_channels//2, 
                                                out_channels= out_channels * self.out_repres_dim * self.out_repres_dim * in_channels,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, x):
        batch_size = x.shape[0]

        z_1 = self.conv_1(x)
        z_1 = F.relu(z_1)
        #print("z_1", z_1.shape)

        z_2 = self.conv_2(z_1)
        z_2 = F.relu(z_2)
        #print("z_2", z_2.shape)

        z_2_shape = z_2.shape
        z_q = z_2.permute(0, 2, 3, 1).contiguous()
        z_q = z_q.view(-1, z_2_shape[2] * z_2_shape[3], z_2_shape[1])

        z_q = self.hopfield(z_q)
        z_q = z_q.view(-1, z_2_shape[2], z_2_shape[3], z_2_shape[1])
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        q_error = self.q_loss(z_2, z_q)
        #print("q_error", q_error.shape)

        q_x_jacob = jacobian(self.propagate, x, create_graph=True)
        '''
        # batch_size, 1, out_repres_dim ** 2, out_channels, out_repres_dim ** 2
        e_q_jacob = self.e_q(q_error)
        e_q_jacob = e_q_jacob.view(batch_size, 1, self.out_repres_dim ** 2, self.out_channels, self.out_repres_dim ** 2)
        #print("e_q_jacob", e_q_jacob.shape)

        q_z_2_jacob = self.q_z_2(z_q)
        q_z_2_jacob = F.relu(q_z_2_jacob)
        #print("q_z_2", q_z_2_jacob.shape)

        q_z_1_jacob = self.q_z_1(q_z_2_jacob + z_2)
        q_z_1_jacob = F.relu(q_z_1_jacob)
        #print("q_z_1", q_z_1_jacob.shape)

        # batch_size, out_channels, out_repres_dim ** 2, in_channels, in_repres_dim ** 2
        q_x_jacob = self.q_x(q_z_1_jacob + z_1)
        q_x_jacob = q_x_jacob.view(batch_size, self.out_channels, self.out_repres_dim ** 2, self.in_channels, self.in_repres_dim ** 2)
        #print("q_x_jacob", q_x_jacob.shape)

        #So want to matmul along the last/first three dimensions
        # batch_size, 1, out_repres_dim ** 2, in_channels, in_repres_dim ** 2
        #e_x_jacob = e_q_jacob.matmul(q_x_jacob)
        e_x_jacob = torch.einsum('abcde,adefg->abcfg', e_q_jacob, q_x_jacob)
        #print("e_x", e_x_jacob.shape)

        # batch_size, 1, out_repres_dim ** 2, 1, in_repres_dim ** 2
        e_x_jacob_sum = e_x_jacob.sum(dim=-2, keepdim=True)
        # batch_size, 1, 1, 1, in_repres_dim ** 2
        e_x_jacob_max, _ = torch.max(e_x_jacob_sum, dim=2, keepdim=True)
        # batch_size, 1, out_repres_dim ** 2, 1, in_repres_dim ** 2
        e_x_jacob_max = e_x_jacob_max.expand_as(e_x_jacob_sum)

        # batch_size, out_channels, out_repres_dim ** 2, in_channels, in_repres_dim ** 2
        jacob_mask = torch.where(e_x_jacob_sum < e_x_jacob_max, 0.0, 1.0).expand_as(q_x_jacob)
        q_x_jacob_masked = torch.where(jacob_mask == 1.0, q_x_jacob, 0.0).permute(0, 3, 4, 1, 2).contiguous()
        '''
        q_x_jacob_masked = q_x_jacob.permute(0, 3, 4, 1, 2).contiguous()

        #z_q_masked = x.matmul(q_x_jacob_masked)
        z_q_masked = torch.einsum('abc,abcde->ade', x.view(batch_size, self.in_channels, self.in_repres_dim ** 2), q_x_jacob_masked)
        z_q_masked = z_q_masked.permute(0, 2, 1).contiguous()
        z_q_masked = self.hopfield(z_q_masked)
        z_q_masked = z_q_masked.permute(0, 2, 1).contiguous()

        return z_q_masked.view(batch_size, self.out_channels, self.out_repres_dim, self.out_repres_dim)

        '''z_q_masked = torch.zeros_like(z_q, requires_grad=True).view(batch_size, self.out_channels, -1)
        # Use scatter on indices like in vq-vae to fill zeros tensor with only single output tensor for each input tensor
        for batch_num in range(batch_size):

            #out_repres_dim, out_repres_dim, in_channels, in_repres_dim, in_repres_dim
            e_x_grad = e_x_jacob[batch_num, :, :, :, :, :].sum(dim=0).squeeze()

            #out_repres_dim * out_repres_dim, in_repres_dim * in_repres_dim
            e_x_total = e_x_grad.sum(dim=2).view(out_repres_dim ** 2, in_repres_dim ** 2)

            #in_repres_dim * in_repres_dim
            _, e_x_max_ind = torch.min(e_x_total, dim=0)

            #out_channels, out_repres_dim, out_repres_dim, in_channels, in_repres_dim, in_repres_dim
            q_x_grad = q_x_jacob[batch_num, :, :, :, :, :, :].squeeze()

            #out_channels, out_repres_dim * out_repres_dim, in_channels, in_repres_dim * in_repres_dim
            q_x_grad = q_x_grad.view(self.out_channels, out_repres_dim ** 2, self.in_channels, in_repres_dim ** 2)
            
            for ind in range(in_repres_dim ** 2):
                max_ind = e_x_max_ind[ind]
                z_q_masked[batch_num, :, max_ind] += x[batch_num, :, ind].matmul(q_x_grad[:, max_ind, :, ind].T)

        return self.q(z_q_masked.view_as(z_q))
        '''

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

    def forward(self, x):
        y = self.propagate(x)
        y_shape = y.shape
        y = y.permute(0, 2, 3, 1).contiguous()
        y = y.view(y_shape[0], -1, y_shape[1])

        y_q = self.hopfield(y)
        y_q = y_q.view(y_shape[0], y_shape[2], y_shape[3], -1)
        return y_q.permute(0, 3, 1, 2).contiguous()

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        #Best so far 192
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