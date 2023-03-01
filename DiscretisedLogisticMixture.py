import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as dist

import numpy as np

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot

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
                
        # B x T x 1 -> B x T x num_mixtures
        X = X.expand_as(MU)

        centered_X = X - MU
        inv_S = torch.exp(-log_S)

        plus_in = inv_S * (centered_X + (1. / (self.num_levels - 1)))
        cdf_plus = torch.sigmoid(plus_in)

        min_in = inv_S * (centered_X - (1. / (self.num_levels - 1)))
        cdf_min = torch.sigmoid(min_in)

        # log probability for edge case of 0 (before scaling)
        # equivalent: torch.log(F.sigmoid(plus_in))
        log_cdf_plus = plus_in - F.softplus(plus_in)

        # log probability for edge case of 255 (before scaling)
        # equivalent: (1 - F.sigmoid(min_in)).log()
        log_one_minus_cdf_min = -F.softplus(min_in)

        # probability for all other cases
        cdf_delta = cdf_plus - cdf_min

        mid_in = inv_S * centered_X
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in our code)
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
        '''
        # sample mixture indicator from softmax
        temp = logit_PI.data.new(logit_PI.size()).uniform_(1e-5, 1.0 - 1e-5)
        temp = logit_PI.data - torch.log(- torch.log(temp))
        _, argmax = temp.max(dim=-1)

        # (B, T) -> (B, T, nr_mix)
        one_hot = to_one_hot(argmax, self.num_mixtures)
        # select logistic parameters
        MU = torch.sum(MU * one_hot, dim=-1)
        log_scale_min = -32.23619130191664
        log_S = torch.clamp(torch.sum(log_S * one_hot, dim=-1), min=log_scale_min)
        # sample from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = MU.data.new(MU.size()).uniform_(1e-5, 1.0 - 1e-5)
        X = MU + torch.exp(log_S) * (torch.log(u) - torch.log(1. - u))

        X = torch.clamp(torch.clamp(X, min=-1.), max=1.)

        return X.view(-1, self.num_channels, self.image_size, self.image_size)
        '''

        temp = logit_PI.data.new(logit_PI.size()).uniform_(1e-5, 1.0 - 1e-5)
        temp = logit_PI.data - torch.log(- torch.log(temp))
        _, argmax = temp.max(dim=-1)
        sel = to_one_hot(argmax, self.num_mixtures).unsqueeze(3)

        # select logistic parameters
        MU = torch.sum(MU * sel, dim=4)
        log_S = torch.clamp(torch.sum(log_S * sel, dim=4), min=-7.)

        # sample from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = MU.data.new(MU.size()).uniform_(1e-5, 1.0 - 1e-5)
        X = MU + torch.exp(log_S)*(torch.log(u) - torch.log(1. - u))
        X = torch.clamp(torch.clamp(X, min=-1.), max=1.)

        X = X.view(-1, self.image_size, self.image_size, self.num_channels)
        return X.permute(0, 3, 1, 2).contiguous()