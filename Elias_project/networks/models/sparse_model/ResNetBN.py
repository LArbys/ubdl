# Imports
import os,sys,time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# sys.path.append("/home/ebengh01/SparseConvNet")
import sparseconvnet as scn
import se_module as se
# sys.path.append("/home/ebengh01/ubdl/Elias_project/networks/models/dense_model")
sys.path.append("/home/ebengh01/ubdl/larflow/deprecated/sparse_larflow/")
import utils_sparselarflow as res
# import dense_se_module as dse

import time
import math
import numpy as np

class SEResNetBN(nn.Module):
    def __init__(self, dimension, in_channels, out_channels):
    
        nn.Module.__init__(self)
        self.nIn = in_channels
        self.nOut = out_channels
        self.dim = dimension
        
        self.purpleBlock = res.create_resnet_layer(3, self.nIn, self.nOut)

        self.nOut = self.nIn//3
        self.greenBlock = res.create_resnet_layer(4, self.nIn, self.nOut)
        self.nIn = self.nOut
        
        self.nOut = self.nIn//2
        self.orangeBlock = res.create_resnet_layer(6, self.nIn, self.nOut)
        self.nIn = self.nOut

        self.nOut = self.nIn//2
        self.blueBlock = res.create_resnet_layer(3, self.nIn, self.nOut)        
    
    def forward(self, x, inputshape):
        # Purple Block:
        print("initial BN device:",x.features.device)
        tic = time.perf_counter()
        x = self.purpleBlock(x)
        toc = time.perf_counter()
        print(f"purpleBlock in {toc - tic:0.4f} seconds")
        print("post purpleBlock:",x.features.device)
        
        inputshape[0] = inputshape[0]//2 + 1
        inputshape[1] = inputshape[1]//2 + 1
        
        tic = time.perf_counter()
        x = self.greenBlock(x)
        toc = time.perf_counter()
        print(f"greenBlock in {toc - tic:0.4f} seconds")
        print("post greenBlock:",x.features.device)

        inputshape[0] = inputshape[0]//2 + 1
        inputshape[1] = inputshape[1]//2 + 1
        tic = time.perf_counter()
        x = self.orangeBlock(x)
        toc = time.perf_counter()
        print(f"orangeBlock in {toc - tic:0.4f} seconds")
        print("post orangeBlock:",x.features.device)

        inputshape[0] = inputshape[0]//2 + 1
        inputshape[1] = inputshape[1]//2 + 1
        tic = time.perf_counter()
        x = self.blueBlock(x)
        toc = time.perf_counter()
        print(f"blueBlock in {toc - tic:0.4f} seconds")
        print("post blueBlock:",x.features.device)

        return x
    
    
    # def purpleBlock(self, x, inputshape, device):
    #     x = self.purpleBlockP1(x)
    #     x = self.PurpleSE(x, inputshape, device)
    #     x = self.postSE(x)
    #     return x
    
    # def greenTransitionBlock(self, x, inputshape, device):
    #     x = self.greenTransitionBlockP1(x)
    #     x = self.greenTSE(x, inputshape, device)
    #     x = self.postSE(x)
    #     return x
    
    # def greenBlock(self, x, inputshape, device):
    #     x = self.greenBlockP1(x)
    #     x = self.greenSE(x, inputshape, device)
    #     x = self.postSE(x)
    #     return x
    
    # def orangeTransitionBlock(self, x, inputshape, device):
    #     x = self.orangeTransitionBlockP1(x)
    #     x = self.orangeTSE(x, inputshape, device)
    #     x = self.postSE(x)
    #     return x
    
    # def orangeBlock(self, x, inputshape, device):
    #     x = self.orangeBlockP1(x)
    #     x = self.orangeSE(x, inputshape, device)
    #     x = self.postSE(x)
    #     return x
    
    # def blueTransitionBlock(self, x, inputshape, device):
    #     x = self.blueTransitionBlockP1(x)
    #     x = self.blueTSE(x, inputshape, device)
    #     x = self.postSE(x)
    #     return x
    
    # def blueBlock(self, x, inputshape, device):
    #     x = self.blueBlockP1(x)
    #     x = self.blueSE(x, inputshape, device)
    #     x = self.postSE(x)
    #     return x
