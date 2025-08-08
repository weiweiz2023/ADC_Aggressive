import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# Custom blocks (maybe change?)
from src.adc_module import Nbit_ADC
from src.weight_quant import BitSliceQuantSTE, STE_Quantize

class quantized_conv(nn.Module):
    def __init__(self, in_channels, out_channels, arch_args, 
                 kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None):
        super(quantized_conv, self).__init__()
        args = arch_args
        # arch_params order -> [wb, wb/slice, ab, ab/slice, subarray_size, w_stoch_round
        #                       adc_prec, adc_grad_filter, save_adc, adc_stoch_round, 
        #                       adc_pos_only, adc_static_step, acm_fixed_bits, acm_frac_bits  
        #                       adc_custom_loss, stream_init, shared_adc]
        """TODO: ACM implementation for fractional representation"""

        # Weight / Crossbar items
        
        self.w_bits = args.w_bits
        self.w_bits_per_slice = args.w_bits_per_slice
        self.w_slices = int(self.w_bits / max(self.w_bits_per_slice, 1))
        self.wa_stoch_round = args.wa_stoch_round
        subarray_size = args.subarray_size
        if subarray_size <= 0:
            self.num_subarrays = 0
        else:
            self.num_subarrays = self.get_chunks(in_channels * (kernel_size ** 2), subarray_size)
        
        # input vector items
        self.a_bits = args.a_bits
        self.a_bits_per_stream = args.a_bits_per_stream
        self.a_streams = int(self.a_bits / max(self.a_bits_per_stream, 1))
        self.Vmax= args.Vmax
        # ADC items
        self.Goff =args.Goff
        self.Gon = args.Gon
        self.adc_prec = args.adc_prec
        self.adc_grad_filter = args.adc_grad_filter
        self.save_adc_inputs = args.save_adc
        self.adc_stoch_round = args.adc_stoch_round
        self.adc_static_step = args.adc_static_step
        self.adc_pos_only = args.adc_pos_only
        self.adc_custom_loss = args.adc_custom_loss
        
        # Standard convolution params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.bias = bias
        self.kernel_size = kernel_size
    
        # Other
        self.experiment_state = args.experiment_state
        # self.acm_bits = args.acm_fixed_bits
        # self.acm_frac_bits = args.acm_frac_bits
        # self.acm_int_bits = int(self.acm_bits - self.acm_frac_bits)

        self.slice_init = args.slice_init
        if self.slice_init:
            self.weight = nn.Parameter(torch.empty(out_channels  , in_channels, kernel_size, kernel_size))
          #  self.weight = nn.Parameter(torch.empty(out_channels * self.w_slices, in_channels, kernel_size, kernel_size))
        else:
            self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight)

        w_states = 2 ** self.w_bits_per_slice
        a_states = 2 ** self.a_bits_per_stream
        self.shared_adc = args.shared_adc
        if self.shared_adc:
            self.ADC = Nbit_ADC(self.adc_prec, self.w_slices, self.a_streams, self.adc_static_step, self.save_adc_inputs,
                                self.adc_custom_loss, self.adc_stoch_round, self.adc_grad_filter, self.adc_pos_only)
        else:
            # Create an ADC for each partial sum
            self.ADCs = nn.ModulesList(Nbit_ADC(self.adc_prec, self.w_slices, self.a_streams, self.adc_static_step, self.save_adc_inputs,
                                       self.adc_custom_loss, self.adc_stoch_round, self.adc_grad_filter, self.adc_pos_only) for i in range(self.num_subarrays))
        
    @staticmethod
    def get_chunks(in_channels, subarray_size):
        return int(in_channels / subarray_size).__ceil__()
    
    def weight_to_diff_conductance(self, weight, Gon=1, Goff=1/1000):
        w_bits_per_slice = self.w_bits_per_slice
        Nstates_slice = 2**w_bits_per_slice-1
        W_pos = weight.clamp(min=0)   
        W_neg = abs(weight.clamp(max=0))  
        
        G_pos =( W_pos*(Gon - Goff) / Nstates_slice + Goff )        
        G_neg = ( W_neg*(Gon - Goff) / Nstates_slice + Goff )
        
        pos_dummy=G_pos.clone()
        neg_dummy=G_neg.clone()
        with torch.no_grad():
                  pos_dummy .fill_(1.0)  
                  neg_dummy .fill_(1.0)
        G_pos_dummy =  pos_dummy *  (Goff)
        G_neg_dummy =  neg_dummy *  (Goff)
        
        return G_pos, G_neg,G_pos_dummy, G_neg_dummy
  
    def inference_conv(self, inputs, weights):
        a_bits_per_stream = self.a_bits_per_stream
        w_bits_per_slice = self.w_bits_per_slice
        Vmax = self.Vmax
        Gon = self.Gon
        Goff = self.Goff
      #  Comp_factor =a_bits_per_stream*w_bits_per_slice/((Gon-Goff)*Vmax)
        G_pos, G_neg,G_pos_dummy, G_neg_dummy = self.weight_to_diff_conductance(weights, Gon=1, Goff=1/1000)   
        image_map = F.unfold(inputs, self.kernel_size, self.dilation, 
                         self.padding, self.stride)  # [batch, in_ch*K*K, H_out*W_out]       
        V_real = image_map * Vmax / a_bits_per_stream 
        kernel_list = torch.chunk(V_real, chunks=self.num_subarrays, dim=1)  # List of [batch, sub_ch*K*K, H_out*W_out]
        G_pos_list = torch.chunk(G_pos.flatten(1), chunks=self.num_subarrays, dim=1)  # List of [out_ch, sub_ch*K*K]
        G_neg_list = torch.chunk(G_neg.flatten(1), chunks=self.num_subarrays, dim=1)
        G_pos_dummy_list = torch.chunk(G_pos_dummy.flatten(1), chunks=self.num_subarrays, dim=1)
        G_neg_dummy_list = torch.chunk(G_neg_dummy .flatten(1), chunks=self.num_subarrays, dim=1)

        accum_out = 0
        accum_loss = 0
        for i, (G_pos_sub, G_neg_sub,G_pos_dummy_sub,G_neg_dummy_sub ) in enumerate(zip(G_pos_list, G_neg_list,G_pos_dummy_list, G_neg_dummy_list)):

            working_kernel = kernel_list[i].transpose(-2, -1)
        # Analog MVM: I_out = V * G+ - V * G-
            I_pos_inter = F.linear(working_kernel, G_pos_sub)
            I_pos_dummy = F.linear(working_kernel, G_pos_dummy_sub)    # [batch, H_out*W_out, out_ch]
            I_pos = (I_pos_inter - I_pos_dummy)
            I_neg_inter = F.linear(working_kernel, G_neg_sub)
            I_neg_dummy = F.linear(working_kernel, G_neg_dummy_sub)
            I_neg = (I_neg_inter - I_neg_dummy)  # [batch, H_out*W_out, out_ch]
            I_out = (I_pos - I_neg).transpose(-2, -1)  # [batch, out_ch, H_out*W_out]
        # Quantize with ADC
            if self.shared_adc:
                sub_out, loss = self.ADC(I_out)  # Shared ADC
            else:
                sub_out, loss = self.ADCs[i](I_out)  # Independent ADC per subarray

            accum_out += sub_out
            accum_loss += loss
        recover_out = accum_out *1 #Comp_factor  
        out_pixels = int((recover_out.size(dim=-1)) ** 0.5)
        result = F.fold(recover_out, (out_pixels, out_pixels), (1, 1))  # [batch, out_ch, H_out, W_out]
        return result, accum_loss
    
    def partial_sum_conv(self, inputs, weights):
        image_map = F.unfold(inputs, self.kernel_size, self.dilation, self.padding, self.stride)
        flattened_weights = torch.flatten(weights, 1)
        kernel_list = torch.chunk(image_map, chunks=self.num_subarrays, dim=1) # input to each subarray
        weight_list = torch.chunk(flattened_weights, chunks=self.num_subarrays, dim=1) # weight of each subarray

        accum_out = 0
        accum_loss = 0
        for i, working_weight in enumerate(weight_list):
            working_kernel = kernel_list[i].transpose(-2, -1)
            linear_temp = F.linear(working_kernel, working_weight).transpose(-2, -1)
            if self.shared_adc:
                sub_out, loss = self.ADC(linear_temp)
            else:
                sub_out, loss = self.ADCs[i](linear_temp)
            accum_out = accum_out + sub_out
            accum_loss = accum_loss + loss
            
        # Handle weight slice positional scaling
        accum_out = torch.stack(torch.split(accum_out, self.out_channels, 1), -1)
        scalar_vector = 2 ** torch.arange(0, self.w_bits, device="cuda") / (2 ** self.w_bits - 1)
        accum_out = (accum_out * scalar_vector).sum(-1)
        
        out_pixels = int((accum_out.size(dim=-1)) ** 0.5)
        result = F.fold(accum_out, (out_pixels, out_pixels), (1, 1))
        return result, accum_loss 

    def forward(self, inputs):
        # Size = [out_channels, in_channels, kernel_height, kernel_width]
        """TODO: Discuss the tensor dims of weights and how it changes with slice value"""
        qw = self.weight
        if self.w_bits != 0:
            if self.slice_init:
                qw = STE_Quantize.apply(self.weight, self.w_bits_per_slice)
                """TODO: add stoch round? maybe as param to STE_quant?"""
            else:
                qw = BitSliceQuantSTE.apply(self.weight, self.w_bits)

        # Size = [batch_size, in_channels * kern_h * kern_w, pixel_height * pixel_width]
        """TODO: support non-zero bit-stream for training (currently impossible?)"""
        qa = inputs
        if self.a_bits != 0:  # 0 means infinite prec (float)
            qa = STE_Quantize().apply(inputs, self.a_bits)
            """TODO: add stoch round? maybe as param to STE_quant?"""

        if self.experiment_state == "inference" and self.num_subarrays > 0:
            output, a_loss = self.inference_conv(qa, qw)     
        elif self.experiment_state == "PTQAT" and self.num_subarrays > 0:  # Indicate use of reg conv2d (no subarrays)
            output, a_loss = self.partial_sum_conv(qa, qw)#[128,16,28,28]  [64,16,3,3]
        else:
            conv_out = F.conv2d(qa, qw, bias=None, stride=self.stride, padding=self.padding, 
                            dilation=self.dilation, groups=self.groups)
            output, a_loss = self.ADC(conv_out)
        
        return output, a_loss