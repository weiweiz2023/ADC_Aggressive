import torch
from torch.autograd import Function



def int_to_sliced_binary(int_repr, n_bits):
    mask = 1 << torch.arange(n_bits, device=int_repr.device).flip(0)  # 高位在前
    bits = int_repr.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    return bits.reshape(int_repr.shape[0], -1)

class BitSliceQuantSTE(Function):
    @staticmethod
    def forward(ctx, weight, weight_bits):
        ctx.save_for_backward(weight)
        ctx.weight_bits = weight_bits
        max_int = 2 ** weight_bits - 1
        int_repr = torch.round(weight * max_int).int()
        binary = int_to_sliced_binary(int_repr, weight_bits)
        return binary

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        weight_bits = ctx.weight_bits
        
       
        grad_reshaped = grad_output.reshape(weight.shape[0], -1, weight_bits)
        

        bit_weights = torch.tensor([2 ** (weight_bits - 1 - i) for i in range(weight_bits)],
                                 device=grad_output.device, dtype=torch.float32)
        
        grad_weight = (grad_reshaped * bit_weights).sum(dim=-1)
        
        return grad_weight, None


class STE_Quantize(Function):
    @staticmethod
    def forward(ctx, input_tens, bits):
        ctx.save_for_backward(input_tens)
        if bits > 1:
            neg_2s = 2 ** (bits - 1)
            pos_2s = neg_2s - 1
            if pos_2s == 0:  # 1-bit -> 1.58-bit
                pos_2s = 1
            negative_tensor = torch.where(input_tens < 0, input_tens, 0)
            positive_tensor = torch.where(input_tens > 0, input_tens, 0)
            negative_tensor = torch.round(negative_tensor * neg_2s) / neg_2s
            positive_tensor = torch.round(positive_tensor * pos_2s) / pos_2s
            out = (positive_tensor + negative_tensor)
        elif bits == 1:
            out = torch.where(input_tens < 0, torch.floor(input_tens), input_tens)
            out = torch.where(out > 0, torch.ceil(out), out)
        else:
            raise ValueError("Weight bits cannot be <= 0")

        out = torch.clamp(out, -1, 1)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_tens, = ctx.saved_tensors
        
        # Clamp ends (-3, 3) to avoid exploding gradients
        grad_out = torch.where(input_tens > -1, grad_output, 0)
        grad_out = torch.where(input_tens < 1, grad_out, 0)
        
        # Implement straight through estimator (STE)
        # grad_out = grad_output
        return grad_out, None