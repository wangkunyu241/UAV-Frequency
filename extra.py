import math
import torch
import numpy as np
from torch.nn import functional as F
from torch._jit_internal import Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import nn
from torch.nn import init
import torch.fft

import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import sys
from copy import deepcopy
import random
from torch.optim import lr_scheduler
from torchvision.transforms import Resize
from PIL import Image
from fvcore.nn import FlopCountAnalysis, parameter_count_table

class Filter_One_Side(nn.Module):
    def __init__(self, C=3, H=600, W=1138):
        super(Filter_One_Side, self).__init__()
        self.C = C
        self.H = H
        self.W = W
        self.learnable_h = self.H
        self.learnable_w = np.floor(self.W/2).astype(int) + 1
        self.register_parameter('conv_invariant' , torch.nn.Parameter(torch.rand(self.C, self.learnable_h, self.learnable_w), requires_grad=True))
    def forward(self, feature):
        feature_fft = torch.fft.rfftn(feature, dim=(-2, -1))
        feature_fft = feature_fft + 1e-8
        feature_amp = torch.abs(feature_fft)
        feature_pha = torch.angle(feature_fft)
        feature_amp_invariant = torch.mul(feature_amp, self.conv_invariant)
        feature_fft_invariant = feature_amp_invariant * torch.exp(torch.tensor(1j) * feature_pha)
        feature_invariant = torch.fft.irfftn(feature_fft_invariant, dim=(-2, -1) )
        return feature_invariant, feature-feature_invariant


class ContrastiveMLP(nn.Module):
    def __init__(self, in_channel=2048, out_channel=128):
        super(ContrastiveMLP, self).__init__()

    def forward(self, x_invariant, x_specific):
        x_invariant = F.adaptive_avg_pool2d(x_invariant, (1,1))
        x_invariant = x_invariant.view(x_invariant.size(0), -1)

        x_specific = F.adaptive_avg_pool2d(x_specific, (1,1))
        x_specific = x_specific.view(x_specific.size(0), -1)
        return x_invariant, x_specific