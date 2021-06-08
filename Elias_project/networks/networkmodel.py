# Imports
import os,sys,time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append("/home/ebengh01/SparseConvNet")
import sparseconvnet as scn

import time
import math
import numpy as np

class SparseClassifier(nn.Module):
    def __init__(self, inputshape, reps, nin_features, nout_features, nplanes,show_sizes):
        nn.Module.__init__(self)
        """
        inputs
        ------
        inputshape [list of int]: dimensions of the matrix or image
        reps [int]: number of residual modules per layer (for both encoder and decoder)
        nin_features [int]: number of features in the first convolutional layer
        nout_features [int]: number of features that feed into the regression layer
        nPlanes [int]: the depth of the U-Net
        show_sizes [bool]: if True, print sizes while running forward
        """
        self._mode = 0
        self._dimension = 2
        self._inputshape = inputshape
        if len(self._inputshape)!=self._dimension:
            raise ValueError("expected inputshape to contain size of 2 dimensions only."
                             +"given %d values"%(len(self._inputshape)))
        self._reps = reps
        self._nin_features = nin_features
        self._nout_features = nout_features
        self._nplanes = [nin_features, 2*nin_features, 3*nin_features, 4*nin_features, 5*nin_features]
        self._layers = [['s',16,reps,1]]
        self._show_sizes = show_sizes

        # self.sparseModel = scn.Sequential().add(
        #    scn.InputLayer(self._dimension, self._inputshape, mode=self._mode)).add(
        #    scn.SubmanifoldConvolution(self._dimension, 1, self._nin_features, 3, False)).add(
        #    scn.UNet(self._dimension, self._reps, self._nplanes, residual_blocks=True, downsample=[2,2])).add(
        #    scn.BatchNormReLU(self._nin_features)).add(
        #    scn.OutputLayer(self._dimension))
        
        

        self.input = scn.InputLayer(self._dimension, self._inputshape, mode=self._mode)
        self.conv1 = scn.SubmanifoldConvolution(self._dimension, 1, self._nin_features, 7, False) # change to 65 convolution kernels? how?
        self.maxPool = scn.MaxPooling(self._dimension,2,2) # Unsure if this is the right pool_size and pool_stride
        self.sparseSEResNetB2 = self.SparseSEResNetB2(self._dimension, self._nin_features, self._layers)
        # self.testSE = self.SELayer(self._nin_features)
        # self.sparseResNet = scn.SparseResNet(self._dimension, self._nin_features, self._layers)
        self.batchnorm = scn.BatchNormReLU(self._nin_features)
        self.output = scn.OutputLayer(self._dimension)
        self.conv2 = scn.SubmanifoldConvolution(self._dimension, self._nin_features, 1, 3, False)

    def forward(self,coord_t,input_t,batchsize):
        if self._show_sizes:
            print( "coord_t ",coord_t.shape)
            print( "input_t ",input_t.shape)
        x=(coord_t,input_t,batchsize)
        x=self.input(x)
        if self._show_sizes:
            print ("inputlayer: ",x.features.shape)
        x=self.conv1(x)
        if self._show_sizes:
            print ("conv1: ",x.features.shape)
        x=self.sparseResNet(x)
        if self._show_sizes:
            print ("unet: ",x.features.shape)
        x=self.batchnorm(x)
        if self._show_sizes:
            print ("batchnorm: ",x.features.shape)
        x=self.conv2(x)
        if self._show_sizes:
            print ("conv2: ",x.features.shape)
        x=self.output(x)
        if self._show_sizes:
            print ("output: ",x.shape)
        return x

        
    def SparseSEResNetB2(self,dimension, nInputPlanes, layers):
        """
        pre-activated ResNet
        e.g. layers = {{'basic',16,2,1},{'basic',32,2}}
        """
        nPlanes = nInputPlanes
        m = scn.Sequential()

        for blockType, n, reps, stride in layers:
            for rep in range(reps):
                if blockType[0] == 's':  # SE block
                    m.add(scn.BatchNormReLU(nPlanes))
                    m.add(
                        scn.ConcatTable().add(
                            scn.Sequential().add(
                                scn.SubmanifoldConvolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    False) if stride == 1 else scn.Convolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    stride,
                                    False)).add(
                                scn.BatchNormReLU(n)) .add(
                                scn.SubmanifoldConvolution(
                                    dimension,
                                    n,
                                    n,
                                    3,
                                    False)) .add(
                                self.SELayer(nPlanes))) .add(
                                scn.Identity()))
                nPlanes = n
                m.add(scn.AddTable())
        m.add(scn.BatchNormReLU(nPlanes))
        return m

        # TODO: scale the input conv layer with the SE output
    def SELayer(self, channel, reduction=16):
        m = scn.Sequential()
        m.add(scn.SparseToDense(self._dimension, self._nin_features))
        m.add(nn.AdaptiveAvgPool2d(1))
        m.add(scn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), # add back // reduction on output
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False), # add back // reduction on input
            nn.Sigmoid()))
        m.add(scn.DenseToSparse(self._dimension))
        return m
