import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from src.conv_mvm import quantized_conv


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class prune(nn.Module):
    def __init__(self, pruning_rate):
        super(prune, self).__init__()
        self.pruning_rate = pruning_rate
    
    def forward(self, x):
        """Prune the input tensor by setting large values to zero"""
        if self.pruning_rate > 0 and isinstance(x, torch.Tensor):
            k = int( self.pruning_rate * x.numel())
            if k > 0:
                threshold = torch.topk(torch.abs(x).view(-1), k, largest=False)[0][-1]
                mask = torch.abs(x) > threshold
                x = x * mask.float()
        return x


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

 
class BasicBlock_Quant(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, arch_args, stride=1):
        super(BasicBlock_Quant, self).__init__()
        # self.experiment_state= arch_args.experiment_state
        # if self.experiment_state == "pruning"or self.experiment_state == "pretraining":
        #     self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #     self.bn1 = nn.BatchNorm2d(planes)
        #     self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #     self.bn2 = nn.BatchNorm2d(planes)
        #     self.shortcut = nn.Sequential()
        # else: # PTQAT and inference
        self.conv1 = quantized_conv(in_planes, planes, arch_args, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = quantized_conv(planes, planes, arch_args, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        
        if arch_args.pruning:
            self.conv1 = nn.Sequential(prune(arch_args.conv_prune_rate), self.conv1)
            self.conv2 = nn.Sequential(prune(arch_args.conv_prune_rate), self.conv2)
        self.experiment_state= arch_args.experiment_state   
        self.linear_prune_rate = arch_args.linear_prune_rate
        self.conv_prune_rate = arch_args.conv_prune_rate  
        if stride != 1 or in_planes != planes:           
            self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (planes )//4, (planes )//4),
                                                      "constant", 0))
        
    def forward(self, input):   
        x = input[0]
        L0 = input[1]
        
        out, L1 = self.conv1(x)
        out = self.bn1(out)
        shortcut_out = self.shortcut(x) 
        out += shortcut_out
        out = x1 = F.leaky_relu(out)
        
        out, L2 = self.conv2(out)
        out = self.bn2(out)
        out = out + x1
        out = F.leaky_relu(out)
    
        return [out, L0 + L1 + L2]


class ResNet(nn.Module):
    def __init__(self, num_blocks, in_channels, arch_args, start_chan, num_classes=10, block=BasicBlock_Quant):
        super(ResNet, self).__init__()
        self.linear_prune_rate = arch_args.linear_prune_rate
        self.in_planes = start_chan
        self.experiment_state= arch_args.experiment_state 
        self.conv1 = nn.Conv2d(in_channels, start_chan, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(start_chan)
        self.layer1 = self._make_layer(block, start_chan, num_blocks[0], arch_args, stride=1)
        self.layer2 = self._make_layer(block, start_chan * 2, num_blocks[1], arch_args, stride=2)
        self.layer3 = self._make_layer(block, start_chan * 4, num_blocks[2], arch_args, stride=2)
        self.layer4 = self._make_layer(block, start_chan * 8, num_blocks[2], arch_args, stride=2)
        self.bn2 = nn.BatchNorm1d(start_chan * 2 ** (len(num_blocks) ))
        self.linear = nn.Linear(start_chan * 2 ** (len(num_blocks) ), num_classes)
        #self.linear = nn.Linear(start_chan*8 * block.expansion, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, arch_args, stride):
        strides = [stride] + [1]*(num_blocks-1) # 1 1 1 2 1 1 2 1 1
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, arch_args, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out)
        [out, L1] = self.layer1([out, 0])
        [out, L2] = self.layer2([out, L1])
        [out, L3] = self.layer3([out, L2])  
        [out, L4] = self.layer4([out, L3])################
        out = F.avg_pool2d(out, out.size()[3])  # change to max pool?
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        if self.experiment_state == "pruning":
            out = prune(out,self.linear_prune_rate)
        out = self.linear(out)
        return out, L4 * 0.1#######################