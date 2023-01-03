import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
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
        return F.gelu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

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
        x = F.gelu(x)
        
        x = self._conv_2(x)
        x = F.gelu(x)
        
        x = self._conv_3(x)
        x = F.gelu(x)

        x = self._conv_4(x)
        #Should have 2048 units -> embedding_dim * repres_dim^2
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
                                                kernel_size=3, 
                                                stride=3)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=3, 
                                                stride=1, padding=2)

        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=out_channels,
                                                kernel_size=3, 
                                                stride=1, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.gelu(x)

        x = self._conv_trans_2(x)
        x = F.gelu(x)
        
        return self._conv_trans_3(x)

class VAE(nn.Module):
    def __init__(self, config, device):
        super(VAE, self).__init__()

        self.device = device

        self._embedding_dim = config.embedding_dim
        self._latent_dim = config.latent_dim
        self._representation_dim = config.representation_dim

        self._encoder = Encoder(config.num_channels, config.num_hiddens,
                                config.num_residual_layers, 
                                config.num_residual_hiddens)

        self._pre_sample = nn.Conv2d(in_channels=config.num_hiddens, 
                                      out_channels=config.num_filters,
                                      kernel_size=1, 
                                      stride=1)

        self.mu = nn.Linear(config.embedding_dim * config.representation_dim ** 2, config.latent_dim)
        self.log_var = nn.Linear(config.embedding_dim * config.representation_dim ** 2, config.latent_dim)

        self.pre_decode = nn.Linear(config.latent_dim, config.embedding_dim * config.representation_dim ** 2)

        self._decoder = Decoder(config.num_filters,
                            config.num_channels,
                            config.num_hiddens, 
                            config.num_residual_layers, 
                            config.num_residual_hiddens
                        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self):
        mu = torch.randn(1, self._latent_dim).to(self.device)
        log_var = torch.randn(1, self._latent_dim).to(self.device)

        flat_z_sampled = self.reparameterize(mu, log_var)
        flat_z_sampled = self.pre_decode(flat_z_sampled)

        z_sampled = flat_z_sampled.view(1, self._embedding_dim, self._representation_dim, self._representation_dim)

        x_sample = self._decoder(z_sampled)

        return x_sample

    def interpolate(self, x, y):
        if (x.size() == y.size()):
            zx = self._encoder(x)
            zx = self._pre_sample(zx)

            zy = self._encoder(y)
            zy = self._pre_sample(zy)

            z_shape = zy.shape
            flat_zy = zy.view(z_shape[0], -1)
            flat_zx = zx.view(z_shape[0], -1)

            flat_z = (flat_zx + flat_zy) / 2
            mu = self.mu(flat_z)
            log_var = self.log_var(flat_z)

            flat_z_sampled = self.reparameterize(mu, log_var)
            flat_z_sampled = self.pre_decode(flat_z_sampled)

            z_sampled = flat_z_sampled.view(z_shape)

            xy_recon = self._decoder(z_sampled)

            return xy_recon
        return x

    def reconstruct(self, x):
            return self.forward(x)

    def forward(self, x):
        z = self._encoder(x)
        z = F.gelu(self._pre_sample(z))

        z_shape = z.shape

        flat_z = z.view(z_shape[0], -1)

        mu = self.mu(flat_z)
        log_var = self.log_var(flat_z)

        flat_z_sampled = self.reparameterize(mu, log_var)
        flat_z_sampled = self.pre_decode(flat_z_sampled)

        z_sampled = flat_z_sampled.view(z_shape[0], z_shape[1], self._representation_dim, self._representation_dim)

        x_recon = self._decoder(z_sampled)

        return x_recon, mu, log_var