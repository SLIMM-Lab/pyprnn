#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:34:27 2023

@author: malvesmaia
"""
from torch import nn
import torch

# ------------------- Definition of customized layers ------------------------
    
class softLayer(nn.Module): 
    from torch import Tensor
    
    # This layer applies the softplus function to its weights
    # so that all weights associated to it are stictly positive
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(softLayer, self).__init__()
    
        self.in_features = in_features
        self.out_features = out_features
        self.sp = nn.Softplus()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        import math 
    
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
      return nn.functional.linear(input, self.sp(self.weight), self.bias) 