import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd.functional import jacobian

import numpy as np

from hflayers import HopfieldLayer

from utils import get_prior, straight_through_round

class Residual(nn.Module):
    def __init__(self, in_channels, representation_dim, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            CompleteConv1d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      in_representation_dim=representation_dim,
                      out_representation_dim=representation_dim),
            nn.ReLU(True),
            CompleteConv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      in_representation_dim=representation_dim,
                      out_representation_dim=representation_dim),
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, representation_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, representation_dim, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class CompleteConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, in_representation_dim, out_representation_dim, num_embeddings=512):
        super(CompleteConv1d, self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(in_representation_dim, out_representation_dim) for _ in range(in_channels)])
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.hopfield = HopfieldLayer(
                            input_size=out_channels,                           # R
                            quantity=num_embeddings,                             # W_K
                            stored_pattern_as_static=True,
                            state_pattern_as_static=True
                        )
        self.q_loss = lambda x, y: ((x - y.detach()) ** 2).sum(dim=1)

    def propagate(self, x):
        linear_outs = []
        for layer_idx in range(len(self.linear_layers)):
            linear_outs.append(self.linear_layers[layer_idx](x[:, layer_idx, :]))

        linear_outs = torch.stack(linear_outs, dim=1)

        #conv_outs = self.conv(linear_outs)
        #conv_outs = conv_outs.permute(0, 2, 1).contiguous()

        #q_conv_outs = self.hopfield(conv_outs)
        #q_conv_outs = q_conv_outs.permute(0, 2, 1).contiguous()

        return linear_outs

    def forward(self, x):
        #batch_size, in_channels, in_r_dim
        #Feed in batch_size * in_channels, in_repres_dim -> outputs * batch_size, in_repres_dim, out_repres_dim
        #Should give tensor of dimension 
        # Better to propagate -> get conv_outs and q_conv_outs -> get jacobian for mse -> get 
        # Could just take it that which contributes most to quantised vector (not diff) is most relevant/key
        # Subtract this from q_conv_outs
        # batch_num, in_channels, in_r_dim -> (batch_num, out_channels, out_r_dim, in_channels, in_r_dim), (batch_num, out_channels, out_r_dim)

        y = self.propagate(x)
        y_neg = y.clone()

        print("pre_jac")
        xy_grad = jacobian(self.propagate, x).squeeze(dim=0).squeeze(dim=2)
        print("post_jac")
        xy_grad_reduced = xy_grad.sum(dim=0).sum(dim=1)

        _, xy_grad_max_ind = xy_grad_reduced.max(dim=0, keepdim=True)
        xy_grad_max_ind = xy_grad_max_ind.squeeze()

        for index in range(len(xy_grad_max_ind)):
            node_num = xy_grad_max_ind[index]
            for filter_num in range(x.size(1)):
                y_neg[0, filter_num, node_num] -= xy_grad[filter_num, node_num, filter_num, index].squeeze() * x[0, filter_num, index]

        return self.conv(y - y_neg)

class Encoder(nn.Module):
    def __init__(self, in_channels, in_representation_dim, out_representation_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self.conv_1 = CompleteConv1d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 in_representation_dim=in_representation_dim,
                                 out_representation_dim=in_representation_dim // 2)

        self.conv_2 = CompleteConv1d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 in_representation_dim=in_representation_dim // 2,
                                 out_representation_dim=out_representation_dim)


        self.conv_3 = CompleteConv1d(in_channels=num_hiddens,
                                 out_channels=out_representation_dim,
                                 in_representation_dim=out_representation_dim,
                                 out_representation_dim=out_representation_dim)
        '''
        self.conv_4 = CompleteConv1d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 in_representation_dim=out_representation_dim,
                                 out_representation_dim=out_representation_dim)


        self.residual_stack = ResidualStack(in_channels=num_hiddens,
                                             representation_dim=out_representation_dim,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        '''

    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = F.relu(x)
        
        x = self.conv_2(x)
        x = F.relu(x)
        
        x = self.conv_3(x)
        x = F.relu(x)

        #x = self.conv_4(x)
        #Should have 2048 units -> embedding_dim * repres_dim^2
        #return self.residual_stack(x)

        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, in_representation_dim, out_representation_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self.conv_1 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, stride=1)
        '''
        self.residual_stack = ResidualStack(in_channels=num_hiddens,
                                             representation_dim=in_representation_dim,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self.conv_trans_1 = CompleteConv1d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                in_representation_dim=in_representation_dim,
                                                out_representation_dim=in_representation_dim)
        '''


        self.conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=15, stride=1)



        self.conv_trans_3 = nn.ConvTranspose1d(in_channels=num_hiddens//2, 
                                                out_channels=out_channels,
                                                kernel_size=15, stride=1)



    def forward(self, inputs):
        x = self.conv_1(inputs)
        
        #x = self.residual_stack(x)
        
        #x = self.conv_trans_1(x)
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

        self.encoder = Encoder(config.num_channels, config.image_size**2, config.representation_dim**2, config.num_hiddens,
                                config.num_residual_layers, 
                                config.num_residual_hiddens)

        self.pre_vq_conv = CompleteConv1d(in_channels=config.num_hiddens, 
                                      out_channels=config.embedding_dim,
                                      in_representation_dim=config.representation_dim**2,
                                      out_representation_dim=config.representation_dim**2)

        self.hopfield = HopfieldLayer(
                            input_size=config.embedding_dim,                           # R
                            quantity=config.num_embeddings,                             # W_K
                            stored_pattern_as_static=True,
                            state_pattern_as_static=True
                        )

        self.decoder = Decoder(config.embedding_dim,
                        config.num_channels,
                        config.representation_dim**2,
                        config.image_size**2,
                        config.num_hiddens, 
                        config.num_residual_layers, 
                        config.num_residual_hiddens)

    def forward(self, x):
        x = x.view(x.shape[0], self.num_channels, -1)

        z = self.encoder(x)
        z = self.pre_vq_conv(z)

        #64, 64, 8, 8
        z = z.view(-1, self.embedding_dim, self.representation_dim, self.representation_dim)
        z = z.permute(0, 2, 3, 1).contiguous()
        z = z.view(-1, self.representation_dim * self.representation_dim, self.embedding_dim)

        z_embeddings = self.hopfield(z)

        z_embeddings = z_embeddings.view(-1, self.representation_dim, self.representation_dim, self.embedding_dim)
        z_embeddings = z_embeddings.permute(0, 3, 1, 2).contiguous()
        z_embeddings = z_embeddings.view(x.shape[0], self.embedding_dim, -1)

        x_recon = self.decoder(z_embeddings)
        x_recon = x_recon.view(x.shape[0], self.num_channels, self.image_size, self.image_size)
        return x_recon