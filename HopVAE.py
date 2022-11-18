from Residual import ResidualStack

import torch
import torch.nn as nn
import torch.nn.functional as F

from hflayers import HopfieldLayer, Hopfield
from hflayers.transformer import HopfieldEncoderLayer

from Transformer import HopfieldEncoder

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._conv_4 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        x = F.relu(x)

        x = self._conv_4(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=3, 
                                                stride=1, padding=1)

        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=out_channels,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)

        x = self._conv_trans_2(x)
        x = F.relu(x)
        
        return self._conv_trans_3(x)


class HopVAE(nn.Module):
    def __init__(self, config, device):
        super(HopVAE, self).__init__()

        self.device = device
        self.transformer_training = False

        self.attention_dropout = config.attention_dropout 
        self.representation_dim = config.representation_dim

        self.embedding_dim = config.embedding_dim

        self.encoder = Encoder(config.num_channels, config.num_hiddens,
                                config.num_residual_layers, 
                                config.num_residual_hiddens)

        self.pre_vq_conv = nn.Conv2d(in_channels=config.num_hiddens, 
                                      out_channels=config.num_filters,
                                      kernel_size=1, 
                                      stride=1)

        multi_head_attention_block = Hopfield(input_size=config.embedding_dim, num_heads=config.num_heads)
        transformer_block = HopfieldEncoderLayer(multi_head_attention_block)

        self._q_memory = HopfieldLayer(input_size=config.embedding_dim, quantity=config.num_embeddings, stored_pattern_as_static=True, state_pattern_as_static=True)
        self._q_transformer = HopfieldEncoder(encoder_layer=transformer_block, num_layers=config.num_transforms, embedding_dim=config.embedding_dim)

        self.z_memory = HopfieldLayer(input_size=2*config.embedding_dim, quantity=config.num_embeddings, stored_pattern_as_static=True, state_pattern_as_static=True)

        self.z_transformer = HopfieldEncoder(encoder_layer=transformer_block, num_layers=config.num_transforms, embedding_dim=config.embedding_dim)
        self.q_memory = HopfieldLayer(input_size=config.embedding_dim, quantity=config.num_embeddings, stored_pattern_as_static=True, state_pattern_as_static=True)

        self.decoder = Decoder(config.num_filters,
                            config.num_channels,
                            config.num_hiddens, 
                            config.num_residual_layers, 
                            config.num_residual_hiddens
                        )


    def interpolate(self, x, y):
        if (x.size() == y.size()):
            _qx = self.encoder(x)
            _qx = self.pre_vq_conv(_qx)

            _qx = _qx.permute(0, 2, 3, 1).contiguous()
            _qx_shape = _qx.shape

            _qy = self.encoder(y)
            _qy = self.pre_vq_conv(_qy)

            _qy = _qy.permute(0, 2, 3, 1).contiguous()

            _qxy = (_qx + _qy) / 2
            _qxy_vects = _qxy.view(_qx_shape[0], -1, self.embedding_dim)
            _qxy_vects_quantised = self._q_memory(_qxy_vects)

            _q_attention_mask = z_attention_mask = None
            if False:#self.training:
                _q_attention_mask = torch.ones((self.representation_dim**2, self.representation_dim**2)).to(self.device)
                _q_attention_mask = F.dropout(_q_attention_mask, p=1-self.attention_dropout).bool()

                z_attention_mask = torch.ones((self.representation_dim**2, self.representation_dim**2)).to(self.device)
                z_attention_mask = F.dropout(z_attention_mask, p=1-self.attention_dropout).bool()

            if self.transformer_training:
                z_vects = self._q_transformer(_qxy_vects_quantised, mask=_q_attention_mask).contiguous()

                z_vects = z_vects.view(_qx_shape[0], -1, 2*self.embedding_dim)
                z_vects_quantised = self.z_memory(z_vects).contiguous()
                z_vects_quantised = z_vects_quantised.view(_qx_shape[0], -1, self.embedding_dim)

                q_vects = self.z_transformer(z_vects_quantised, mask=z_attention_mask)
                q_vects_quantised = self.q_memory(q_vects).contiguous()
            else:
                q_vects_quantised = _qxy_vects_quantised

            q = q_vects_quantised.view(_qx_shape)
            q = q.permute(0, 3, 1, 2).contiguous()

            xy_recon = self.decoder(q)

            return xy_recon

        return x

    def forward(self, x):
        _q = self.encoder(x)
        _q = self.pre_vq_conv(_q)

        _q = _q.permute(0, 2, 3, 1).contiguous()
        _q_shape = _q.shape
        
        # Flatten input
        _q_vects = _q.view(_q_shape[0], -1, self.embedding_dim)
        _q_vects_quantised = self._q_memory(_q_vects)

        _q_attention_mask = z_attention_mask = None
        if False:#self.training:
            _q_attention_mask = torch.ones((self.representation_dim**2, self.representation_dim**2)).to(self.device)
            _q_attention_mask = F.dropout(_q_attention_mask, p=1-self.attention_dropout).bool()

            z_attention_mask = torch.ones((self.representation_dim**2, self.representation_dim**2)).to(self.device)
            z_attention_mask = F.dropout(z_attention_mask, p=1-self.attention_dropout).bool()

        if self.transformer_training:
            z_vects = self._q_transformer(_q_vects_quantised, mask=_q_attention_mask).contiguous()

            z_vects = z_vects.view(_q_shape[0], -1, 2*self.embedding_dim)
            z_vects_quantised = self.z_memory(z_vects).contiguous()
            z_vects_quantised = z_vects_quantised.view(_q_shape[0], -1, self.embedding_dim)

            q_vects = self.z_transformer(z_vects_quantised, mask=z_attention_mask)
            q_vects_quantised = self.q_memory(q_vects).contiguous()
        else:
            q_vects_quantised = _q_vects_quantised


        q = q_vects_quantised.view(_q_shape)
        q = q.permute(0, 3, 1, 2).contiguous()

        x_recon = self.decoder(q)

        return x_recon