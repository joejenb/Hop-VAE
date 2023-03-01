import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as dist

import numpy as np

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

class DiscretisedLogisticMixture(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.num_mixtures = config.num_mixtures
        self.image_size = config.image_size
        self.num_levels = config.num_levels
        self.num_channels = config.num_channels
        self.batch_size = config.batch_size
        self.device = device

    def forward(self, logit_PI, MU, log_S, X):
        '''
        Args:
            logit_PI: Logits of mixture weightings for each distribution
            MU: Mean values for each distribution
            log_S: Log of scaling parameter for each distribution
            X: Pixel values for which you want to query probabilities of. If not specified it is assumed that all values are
               to be queried. 
        '''
                
        X = X.expand_as(MU)

        centered_X = X - MU
        inv_S = torch.exp(-log_S)

        plus_in = inv_S * (centered_X + (1. / (self.num_levels - 1)))
        cdf_plus = torch.sigmoid(plus_in)

        min_in = inv_S * (centered_X - (1. / (self.num_levels - 1)))
        cdf_min = torch.sigmoid(min_in)

        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = -F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min

        mid_in = inv_S * centered_X
        log_pdf_mid = mid_in - log_S - 2. * F.softplus(mid_in)

        inner_inner_cond = (cdf_delta > 1e-5).float()
        inner_inner_out = inner_inner_cond * \
            torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
            (1. - inner_inner_cond) * (log_pdf_mid - np.log((self.num_levels - 1) / 2))

        inner_cond = (X > 0.999).float()
        inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out

        cond = (X < -0.999).float()
        log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
        log_probs = torch.sum(log_probs, dim=3) + F.log_softmax(logit_PI, dim=-1)

        return -log_sum_exp(log_probs).unsqueeze(-1) 
           
    def sample(self, logit_PI, MU, log_S):
        PI = dist.Categorical(F.softmax(logit_PI.unsqueeze(3), dim=-1))
        S = torch.exp(log_S)

        base_distribution = dist.Uniform(torch.zeros(MU.size()), torch.ones(MU.size()))
        affine_transforms = dist.transforms.AffineTransform(MU, S)

        transforms = [dist.SigmoidTransform().inv, affine_transforms]

        logistic_dists = dist.TransformedDistribution(base_distribution, transforms)
        mixture_logistic = dist.MixtureSameFamily(PI, logistic_dists)

        X = mixture_logistic.sample()
        X = X.view(-1, self.image_size, self.image_size, self.num_channels)
        return X.permute(0, 3, 1, 2).contiguous()