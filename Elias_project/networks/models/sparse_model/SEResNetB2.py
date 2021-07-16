# Imports
import os,sys,time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sparseconvnet as scn
import se_module as se

import time
import math
import numpy as np


class SEResNetB2(nn.Module):
    def __init__(self, dimension, in_channels, out_channels):
        nn.Module.__init__(self)
        self.dim = dimension
        self.nIn = in_channels
        self.nOut = out_channels
        self.preRes = scn.BatchNormReLU(self.nIn)
        self.convA = scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False)
        self.BatchNormReLU = scn.BatchNormReLU(self.nIn)
        self.convB = scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False)
        self.SE = se.SELayer(self.nIn)
        self.postRes = scn.BatchNormReLU(self.nIn)
    
    def forward(self, x, inputshape, device):
        x = self.preRes(x)
        residual = x.features
        x = self.resBlock(x, inputshape, device)
        x.features += residual
        x = self.postRes(x)
        return x        
    
    def resBlock(self, x, inputshape, device):
        x = self.convA(x)
        x = self.BatchNormReLU(x)
        x = self.convB(x)
        x = self.SE(x, inputshape, device)
        return x
