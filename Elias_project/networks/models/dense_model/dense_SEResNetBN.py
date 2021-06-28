# Imports
import os,sys,time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# sys.path.append("/home/ebengh01/SparseConvNet")
import sparseconvnet as scn
import dense_se_module as se

import time
import math
import numpy as np

class SEResNetBN(nn.Module):
    def __init__(self, in_channels, out_channels):# TODO: add back inputshape for sparse
        nn.Module.__init__(self)
        self.nIn = in_channels
        self.nOut = out_channels
        # self.inShape = inputshape
        self.postRes = nn.ReLU()
        self.purpleBlock = nn.Sequential(
                               nn.Conv2d(self.nIn, self.nOut, 3, padding=1, bias=False),
                               nn.BatchNorm2d(self.nIn, eps=0.0001, momentum=0.99),
                               nn.ReLU(),
                               nn.Conv2d(self.nIn, self.nOut, 3, padding=1, bias=False),
                               nn.BatchNorm2d(self.nIn, eps=0.0001, momentum=0.99),
                               se.SELayer(self.nIn), # note, may remove batchnorm for optimization
                               nn.ReLU()
                           )
        self.nOut = self.nIn*2
        self.greenResPad = nn.Conv2d(self.nIn, self.nOut, 1, stride=2, padding=1, bias=False)
        self.greenTransitionBlock = nn.Sequential(
                                        nn.Conv2d(self.nIn, self.nOut, 3, stride=2, padding=2, bias=False),
                                        nn.BatchNorm2d(self.nOut, eps=0.0001, momentum=0.99),
                                        nn.ReLU(),
                                        nn.Conv2d(self.nOut, self.nOut, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(self.nOut, eps=0.0001, momentum=0.99),
                                        se.SELayer(self.nOut), # note, may remove batchnorm for optimization
                                        nn.ReLU()
                                    )
        self.nIn = self.nOut
        self.greenBlock = nn.Sequential(
                               nn.Conv2d(self.nIn, self.nOut, 3, padding=1, bias=False),
                               nn.BatchNorm2d(self.nIn, eps=0.0001, momentum=0.99),
                               nn.ReLU(),
                               nn.Conv2d(self.nIn, self.nOut, 3, padding=1, bias=False),
                               nn.BatchNorm2d(self.nIn, eps=0.0001, momentum=0.99),
                               se.SELayer(self.nIn), # note, may remove batchnorm for optimization
                               nn.ReLU()
                           )
        self.nOut = self.nIn*2
        self.orangeResPad = nn.Conv2d(self.nIn, self.nOut, 1, stride=2, padding=1, bias=False)
        self.orangeTransitionBlock = nn.Sequential(
                                        nn.Conv2d(self.nIn, self.nOut, 3, stride=2, padding=2, bias=False),
                                        nn.BatchNorm2d(self.nOut, eps=0.0001, momentum=0.99),
                                        nn.ReLU(),
                                        nn.Conv2d(self.nOut, self.nOut, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(self.nOut, eps=0.0001, momentum=0.99),
                                        se.SELayer(self.nOut), # note, may remove batchnorm for optimization
                                        nn.ReLU()
                                    )
        self.nIn = self.nOut
        self.orangeBlock = nn.Sequential(
                               nn.Conv2d(self.nIn, self.nOut, 3, padding=1, bias=False),
                               nn.BatchNorm2d(self.nIn, eps=0.0001, momentum=0.99),
                               nn.ReLU(),
                               nn.Conv2d(self.nIn, self.nOut, 3, padding=1, bias=False),
                               nn.BatchNorm2d(self.nIn, eps=0.0001, momentum=0.99),
                               se.SELayer(self.nIn), # note, may remove batchnorm for optimization
                               nn.ReLU()
                           )
        self.nOut = self.nIn*2
        self.blueResPad = nn.Conv2d(self.nIn, self.nOut, 1, stride=2, padding=1, bias=False)
        # self.inShape[0] = self.inShape[0]//2 + 1 # determined by running dense version
        # self.inShape[1] = self.inShape[1]//2 + 1 # determined by running dense version
        self.blueTransitionBlock = nn.Sequential(
                                        nn.Conv2d(self.nIn, self.nOut, 3, stride=2, padding=2, bias=False),
                                        nn.BatchNorm2d(self.nOut, eps=0.0001, momentum=0.99),
                                        nn.ReLU(),
                                        nn.Conv2d(self.nOut, self.nOut, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(self.nOut, eps=0.0001, momentum=0.99),
                                        se.SELayer(self.nOut), # note, may remove batchnorm for optimization
                                        nn.ReLU()
                                    )
        self.nIn = self.nOut
        self.blueBlock = nn.Sequential(
                               nn.Conv2d(self.nIn, self.nOut, 3, padding=1, bias=False),
                               nn.BatchNorm2d(self.nIn, eps=0.0001, momentum=0.99),
                               nn.ReLU(),
                               nn.Conv2d(self.nIn, self.nOut, 3, padding=1, bias=False),
                               nn.BatchNorm2d(self.nIn, eps=0.0001, momentum=0.99),
                               se.SELayer(self.nIn), # note, may remove batchnorm for optimization
                               nn.ReLU()
                           )
    
    
    def forward(self, x):
        for i in range(3):
            res = x
            x = self.purpleBlock(x)
            x += res
            x = self.postRes(x)
            print("purple shape: ",x.shape)
    
        res = self.greenResPad(x)
        x = self.greenTransitionBlock(x)
        x += res
        x = self.postRes(x)
        print("green shape: ",x.shape)
        for i in range(3):
            res = x
            x = self.greenBlock(x)
            x += res
            x = self.postRes(x)
            print("green shape: ",x.shape)
    
        res = self.orangeResPad(x)
        x = self.orangeTransitionBlock(x)
        x += res
        x = self.postRes(x)
        print("orange shape: ",x.shape)
        for i in range(5):
            res = x
            x = self.orangeBlock(x)
            x += res
            x = self.postRes(x)
            print("orange shape: ",x.shape)
    
        res = self.blueResPad(x)
        x = self.blueTransitionBlock(x)
        x += res
        x = self.postRes(x)
        print("blue shape: ",x.shape)
        for i in range(2):
            res = x
            x = self.blueBlock(x)
            x += res
            x = self.postRes(x)
            print("blue shape: ",x.shape)
        return x
