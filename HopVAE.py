import torch
import torch.nn as nn
import torch.nn.functional as F

from HopfieldLayers.hflayers import HopfieldLayer

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
                                                out_channels=out_channels,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)


class HopVAE(nn.Module):
    def __init__(self, config, device):
        super(HopVAE, self).__init__()

        self.device = device

        self._embedding_dim = config.embedding_dim

        self._encoder = Encoder(config.num_channels, config.num_hiddens,
                                config.num_residual_layers, 
                                config.num_residual_hiddens)

        self._pre_vq_conv = nn.Conv2d(in_channels=config.num_hiddens, 
                                      out_channels=config.embedding_dim,
                                      kernel_size=1, 
                                      stride=1)

        self._hopfield = HopfieldLayer(
                            input_size=4096,#config.embedding_dim,                           # R
                            quantity=config.num_embeddings,                             # W_K
                            #scaling=1 / (config.num_embeddings ** (1/2)),
                            stored_pattern_as_static=True,
                            state_pattern_as_static=True
                        )

        self._decoder = Decoder(config.embedding_dim,
                        config.num_channels,
                        config.num_hiddens, 
                        config.num_residual_layers, 
                        config.num_residual_hiddens)

    def forward(self, X):
        Z = self._encoder(X)
        Z = self._pre_vq_conv(Z)

        Z_shape = Z.shape
        flat_Z = Z.view(Z_shape[0], -1)
        flat_Zt = flat_Z.t()

        '''Z_mean = flat_Z.mean(dim=0, keepdim=True)
        Z_centered = flat_Z - Z_mean
        Z_centered_t = Z_centered.t()

        'ZtZ = Z_centered_t.matmul(Z_centered)
        eigVals, eigVecs = torch.linalg.eigh(ZtZ) #the eigvecs are sorted in ascending eigenvalue order.

        V = ZtZ.matmul(eigVecs.detach()) / eigVals.detach()
        Vt = V.t()

        print(eigVals.size())
        print(eigVecs.size())
        print(ZtZ.size())
        print(V.size())
        print(Vt.size())
        Vt_quantised = self._hopfield(Vt.unsqueeze(0)).squeeze()
        V_quantised = Vt_quantised.t()

        flat_Z_quantised = V_quantised.matmul(eigVals).matmul(Vt_quantised) + Z_mean
        '''

        # Want columns of U to be quantised
        U, S, Vt = torch.linalg.svd(flat_Zt, full_matrices=False)
        Ud = flat_Zt.matmul(Vt.t()) / S

        # Try original quantisation but with intermediate svd
        Ud_quantised = self._hopfield(Ud.t().unsqueeze(0)).squeeze(0).t()

        flat_Zt_quantised = Ud_quantised.matmul(torch.diag(S)).matmul(Vt)
        flat_Z_quantised = flat_Zt_quantised.t()

        Z_quantised = flat_Z_quantised.view(Z_shape)

        X_recon = self._decoder(Z_quantised)

        return X_recon