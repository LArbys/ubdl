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
        
        
    # DENSE IMPLEMENTATION:
    # def __init__(self, in_channels, out_channels, reps):
    #     nn.Module.__init__(self)
    #     self.nIn = in_channels
    #     self.nOut = out_channels
    #     self.preRes = nn.Sequential(
    #                       nn.BatchNorm2d(self.nIn, eps=0.0001, momentum=0.99),
    #                       nn.ReLU()
    #                   )
    #     self.resBlock = nn.Sequential(
    #                         nn.Conv2d(self.nIn, self.nOut, 3, padding=1, bias=False),
    #                         nn.BatchNorm2d(self.nIn, eps=0.0001, momentum=0.99),
    #                         nn.ReLU(),
    #                         nn.Conv2d(self.nIn, self.nOut,3, padding=1, bias=False),
    #                         se.SELayer(self.nIn)
    #                     )
    #     self.postRes = nn.Sequential(
    #                        nn.BatchNorm2d(self.nIn, eps=0.0001, momentum=0.99),
    #                        nn.ReLU()
    #                    )
    # 
    # def forward(self, x):
    #     x = self.preRes(x)
    #     residual = x
    #     x = self.resBlock(x)
    #     x += residual
    #     x = self.postRes(x)
    #     return x

    # SPARSE IMPLEMENTATION
    # def __init__(self, dimension, in_channels, out_channels):
    #     nn.Module.__init__(self)
    #     self.dim = dimension
    #     self.nIn = in_channels
    #     self.nOut = out_channels
    #     self.preRes = scn.BatchNormReLU(self.nIn)
    #     self.convA = scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False)
    #     self.BatchNormReLU = scn.BatchNormReLU(self.nIn)
    #     self.convB = scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False)
    #     self.SE = se.SELayer(self.nIn)
    #     # self.resBlock = nn.Sequential(
    #     #                     scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #     #                     scn.BatchNormReLU(self.nIn),
    #     #                     scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #     #                     se.SELayer(self.nIn)
    #     #                 )
    #     self.postRes = scn.BatchNormReLU(self.nIn)
    # 
    # def forward(self, x, inputshape):
    #     print("inputshape in SEResNetB2:",inputshape)
    #     # print("pre-preRes shape: ",x.features.shape)
    #     x = self.preRes(x)
    #     # print("post-preRes shape: ",x.features.shape)
    #     residual = x.features
    #     # print("x type pre resBlock:",type(x))
    #     x = self.resBlock(x, inputshape)
    #     # print("x type post resBlock:",type(x))
    #     # print("post-resBlock shape: ",x.features.shape)
    #     x.features += residual
    #     x = self.postRes(x)
    #     # print("post-postRes shape: ",x.features.shape)
    #     return x        
    # 
    # def resBlock(self, x, inputshape):
    #     x = self.convA(x)
    #     x = self.BatchNormReLU(x)
    #     x = self.convB(x)
    #     x = self.SE(x, inputshape)
    #     return x
