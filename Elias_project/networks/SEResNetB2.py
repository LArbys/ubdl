# Imports
import os,sys,time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# sys.path.append("/home/ebengh01/SparseConvNet")
import sparseconvnet as scn
import se_module as se

import time
import math
import numpy as np


class SEResNetB2(nn.Module):
    def __init__(self, in_channels, out_channels, reps):
        nn.Module.__init__(self)
        self.nIn = in_channels
        self.nOut = out_channels
        self.preRes = nn.Sequential(
                          nn.BatchNorm2d(self.nIn, eps=0.0001, momentum=0.99),
                          nn.ReLU()
                      )
        self.resBlock = nn.Sequential(
                            nn.Conv2d(self.nIn, self.nOut, 3, padding=1, bias=False),
                            nn.BatchNorm2d(self.nIn, eps=0.0001, momentum=0.99),
                            nn.ReLU(),
                            nn.Conv2d(self.nIn, self.nOut,3, padding=1, bias=False),
                            se.SELayer(self.nIn)
                        )
        self.postRes = nn.Sequential(
                           nn.BatchNorm2d(self.nIn, eps=0.0001, momentum=0.99),
                           nn.ReLU()
                       )
                  
    def forward(self, x):
        x = self.preRes(x)
        residual = x
        x = self.resBlock(x)
        x += residual
        x = self.postRes(x)
        return x

