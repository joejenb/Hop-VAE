from Residual import ResidualStack

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VerticalConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, device='cpu', kernel_size=3, dilation=1, mask_center=False, **kwargs):
        super().__init__()
        padding = (dilation * (kernel_size - 1) // 2, dilation * (kernel_size - 1) // 2)

        self.device = device
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), padding=padding, dilation=dilation, **kwargs)
        self.mask = torch.ones(kernel_size, kernel_size).to(self.device)
        self.mask[kernel_size // 2 + 1:, :] = 0
        
        if mask_center:
            self.mask[kernel_size // 2, :] = 0
    
    def forward(self, x):
        self.conv.weight.data *= self.mask.detach()
        return self.conv(x)

class HorizontalConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, device='cpu', dilation=1, mask_center=False, **kwargs):
        super().__init__()
        padding = (0, dilation * (kernel_size - 1) // 2)

        self.device = device
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), padding=padding, dilation=dilation, **kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=padding, dilation=dilation, **kwargs)
        self.mask = torch.ones(1, kernel_size).to(self.device)
        self.mask[0, kernel_size // 2 + 1:] = 0
        
        if mask_center:
            self.mask[0, kernel_size // 2] = 0
    
    def forward(self, x):
        self.conv.weight.data *= self.mask.detach()
        return self.conv(x)

class GatedMaskedConv(nn.Module):
    
    def __init__(self, in_channels, device='cpu', **kwargs):
        super().__init__()
        self.conv_vert = VerticalConv(in_channels, out_channels= 2 * in_channels, device=device, **kwargs)
        self.conv_horiz = HorizontalConv(in_channels, out_channels= 2 * in_channels, device=device, **kwargs)
        self.conv_vert_to_horiz = nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=1, padding=0)
        self.conv_horiz_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
    
    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)
        
        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)

        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)

        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack
        
        return v_stack_out, h_stack_out

class PixelCNN(nn.Module):
    
    def __init__(self, config, device):
        super().__init__()

        self.device = device
        self.num_hiddens = config.num_hiddens
        self.num_channels = config.num_channels
        self.num_categories = config.num_categories
        self.representation_dim = config.representation_dim

        self.conv_vstack = VerticalConv(self.num_channels, self.num_hiddens, device=device, mask_center=True)
        self.conv_hstack = HorizontalConv(self.num_channels, self.num_hiddens, device=device, mask_center=True)

        self.conv_layers = nn.ModuleList([
            GatedMaskedConv(self.num_hiddens, device=device),
            GatedMaskedConv(self.num_hiddens, device=device, dilation=2),
            GatedMaskedConv(self.num_hiddens, device=device),
            GatedMaskedConv(self.num_hiddens, device=device, dilation=4),
            GatedMaskedConv(self.num_hiddens, device=device),
            GatedMaskedConv(self.num_hiddens, device=device, dilation=2),
            GatedMaskedConv(self.num_hiddens, device=device)
        ])

        self.conv_out = nn.Conv2d(self.num_hiddens, self.num_channels * self.num_categories, kernel_size=1, padding=0)
   
    def sample(self):
        img = torch.zeros(1, self.num_channels, self.representation_dim, self.representation_dim).long().to(self.device) - 1

        for h in range(self.representation_dim):
            for w in range(self.representation_dim):
                for c in range(self.num_channels):
                    if (img[:, c, h, w] != -1).all().item():
                        continue

                    pred = self.forward(img[:, :, :h+1, :]) 
                    probs = F.softmax(pred[:, :, c, h, w], dim=-1)
                    img[:, c, h, w] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)

        return img
        
    def interpolate(self, x, y):
        xy_inter = (x + y) / 2
        xy_inter = self.denoise(xy_inter)        

        return xy_inter

    def denoise(self, x):

        pred = self.forward(x)
        pred = pred.permute(0, 2, 3, 4, 1).contiguous()

        pred_shape = pred.shape

        pred = pred.view(-1, self.num_categories)

        probs = F.softmax(pred, dim=-1)

        x_new = torch.multinomial(probs, num_samples=1)
        x_new = x_new.view(pred_shape[0], pred_shape[1], pred_shape[2], pred_shape[3], 1)
        x_new = x_new.permute(0, 4, 1, 2, 3).contiguous().squeeze(1)

        return x_new

    def forward(self, x):
        x = (x.float() / (self.num_categories - 1)) * 2 - 1 

        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)

        out = self.conv_out(F.elu(h_stack))
        
        out = out.reshape(out.shape[0], self.num_categories, self.num_channels, out.shape[2], out.shape[3])
        return out
    