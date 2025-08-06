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
            with open("./saved/hist_csvs/testLossGradNoScale_input.csv", "a") as f:
                array_to_write = (x).flatten().cpu().numpy()
                np.savetxt(f, array_to_write, delimiter=",")
            with open("./saved/hist_csvs/testLossGradNoScale_output.csv", "a") as f:
                array_to_write = (y).flatten().cpu().numpy()
                np.savetxt(f, array_to_write, delimiter=",")
        
        if self.grad_filter:
            gradientFilter.apply(y)

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
    def forward(ctx, input_tens):
        ctx.save_for_backward(input_tens)
        return input_tens
    
    @staticmethod
    def backward(ctx, grad_output):
        input_tens, = ctx.saved_tensors
        grad_out = (.8 * torch.abs(torch.sin(input_tens * torch.pi)) + .2) * grad_output
        
        return grad_out, None