import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def prune_model(model, conv_prune_amount=0.2, linear_prune_amount=0.1):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=conv_prune_amount)
        elif isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=linear_prune_amount)
