import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import os

# For logging purposes
import random


class Nbit_ADC(nn.Module):
    def __init__(self, bits: int, n_state_slices: int, n_state_stream: int, 
    static_step: bool, save_adc: bool, custom_loss: bool, stoch_round: bool,
    grad_filter: bool, pos_only: bool):
        # self.name = number of adcs made or num conv
        super(Nbit_ADC, self).__init__()
        self.bits = bits
        self.save = save_adc
        self.pos_only = pos_only
        self.grad_filter = grad_filter 
        self.custom_loss = custom_loss
        self.stoch_round = stoch_round
        n_slices = n_state_slices
        n_stream = n_state_stream
        self.static_step = static_step
        if self.static_step:
            self.step_size = 1/(n_slices*n_stream)
            self.zero_point = 0
        else:
            self.step_size = nn.Paramter(torch.tensor(1.0))
            self.zero_point = nn.Paramter(torch.tensor(0.00001))

    def forward(self, x):
        scale_offset = x / self.step_size - self.zero_point  

        ## Handle rounding method
        if self.stoch_round: 
            y = stochasticRound.apply(scale_offset)
        else:
            """TODO: Do we rescale by step size? Run tests, no option to be made"""
            y = (torch.round(scale_offset) * self.step_size).detach() + x - x.detach()

        if self.pos_only:
            y = y.clamp(0, 2 ** self.bits - 1)
        else:
            y = y.clamp(-2 ** (self.bits - 1), 2 ** (self.bits - 1) - 1)

        # Do we want a custom loss term accumulated?
        loss = 0    
        if self.custom_loss:
            remainder = x % self.step_size + self.zero_point       
            loss = torch.mean(torch.abs(x - y) ** 2)  # L2 loss
    
        # Logging (histograms)
        if (self.save and (random.random() < 0.003)):  
            print(f"Saving ADC inputs...")
            with open("./saved/hist_csvs/test_nofilter_input.csv", "a") as f:
                array_to_write = (x).flatten().cpu().numpy()
                np.savetxt(f, array_to_write, delimiter=",")
            with open("./saved/hist_csvs/test_nofilter_output.csv", "a") as f:
                array_to_write = (y).flatten().cpu().numpy()
                np.savetxt(f, array_to_write, delimiter=",")
        
        if self.grad_filter:
            y=gradientFilter.apply(y)
            y.requires_grad_(True)
        return y, loss

class stochasticRound(Function):
    @staticmethod
    def forward(ctx, input_tens):
        IMCout_floor = torch.floor(input_tens)
        IMCout_probs = input_tens - IMCout_floor
        IMCout_round = torch.bernoulli(IMCout_probs) + IMCout_floor
        return IMCout_round
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class gradientFilter(Function):
    @staticmethod
    def forward(ctx, input_tens, bits):
        ctx.max_val = 2 ** (bits -1) - 1
        ctx.min_val = -2 ** (bits -1)
        ctx.save_for_backward(input_tens)
        return input_tens
    
    @staticmethod
    def backward(ctx, grad_output):
        input_tens, = ctx.saved_tensors
        scale1 = torch.clamp(0.1 + torch.log(input_tens-ctx.min_val) / 2.5, min=0, max=1.0)
        scale2 = torch.clamp(0.1 + torch.log(-(input_tens+ctx.min_val)) / 2.5, min=0, max=1.0)
        scale3 = torch.clamp(torch.abs(torch.sin(input_tens*torch.pi))*0.9+0.1, min=0, max=1.0)
        
        grad_out = torch.where(input_tens < ctx.min_val, scale2*grad_output, 0)
        grad_out = torch.where(input_tens > ctx.min_val, scale3*grad_output, grad_out)
        grad_out = torch.where(input_tens > ctx.max_val, scale1*grad_output, grad_out)

        return grad_out, None
    


 