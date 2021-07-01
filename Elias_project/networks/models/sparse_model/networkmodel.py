# Imports
import os,sys,time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# sys.path.append("/home/ebengh01/SparseConvNet")
import sparseconvnet as scn
import se_module as se # can get rid of this eventually
import SEResNetB2 as b2
import SEResNetBN as bn
import SparseGlobalAvgPool2d as gavg

import time
import math
import numpy as np

class SparseClassifier(nn.Module):
    def __init__(self, inputshape, nin_features, nout_features, show_sizes):
        nn.Module.__init__(self)
        """
        inputs
        ------
        inputshape [list of int]: dimensions of the image
        nin_features [int]: number of features in the first convolutional layer
        nout_features [int]: number of features that come out of the first
                             convolutional layer
        show_sizes [bool]: if True, print sizes while running forward
        """
        self._mode = 0
        self._dimension = 2
        self._inputshape = inputshape
        if len(self._inputshape)!=self._dimension:
            raise ValueError("expected inputshape to contain size of 2 dimensions only."
                             +"given %d values"%(len(self._inputshape)))
        self._inReps = 3
        self._nin_features = nin_features
        self._nout_features = nout_features
        self._show_sizes = show_sizes        

        self.input = scn.InputLayer(self._dimension, self._inputshape, mode=self._mode)
        self.conv1 = scn.SubmanifoldConvolution(self._dimension, 1, self._nin_features, 7, False)
        self.maxPool = scn.MaxPooling(self._dimension,2,2)
        
        self.inputshape2 = [self._inputshape[0]/2,self._inputshape[1]/2]
        self.SEResNetB2 = b2.SEResNetB2(self._dimension, self._nin_features, self._nout_features)
        self.makeDense = scn.SparseToDense(self._dimension, self._nin_features)
        # concatenate in forward pass
        self.makeSparse = scn.DenseToSparse(self._dimension)
        inputshape3 = self.inputshape2
        inputshape3[0] = 3*self.inputshape2[0]
        self.SEResNetBN = bn.SEResNetBN(self._dimension, self._nin_features, self._nout_features)
        self.makeDense2 = scn.SparseToDense(self._dimension, 512)
        self._nin_features = self._nin_features*8
        self._nout_features = self._nout_features*8
        inputshape4 = [190,218]
        
        self.outputAvg = gavg.SparseGlobalAvgPool2d()
        self.output = nn.Sequential(
                          # nn.Flatten(),
                          nn.Linear(512, 1000)
                      )
        self.flavors = nn.Sequential(
                      nn.Linear(1000, 4),
                      nn.Softmax(0)
                  )
        self.interactionType = nn.Sequential(
                      nn.Linear(1000, 4),
                      nn.Softmax(0)
                  )
        self.nProton = nn.Sequential(
                      nn.Linear(1000, 4),
                      nn.Softmax(0)
                  )
        self.nCPion = nn.Sequential(
                      nn.Linear(1000, 4),
                      nn.Softmax(0)
                  )
        self.nNPion = nn.Sequential(
                      nn.Linear(1000, 4),
                      nn.Softmax(0)
                  )
        self.nNeutron = nn.Sequential(
                      nn.Linear(1000, 4),
                      nn.Softmax(0)
                  )
        
    def forward(self,coord_t,input_t,batchsize):
        print("FORWARD PASS")
        gtic = time.perf_counter()
        # initializes inputshape and shrinks it for use in the SEResNet Block 2
        inputshape = [0,0]
        inputshape[0] = self._inputshape[0]
        inputshape[1] = self._inputshape[1]
        inputshape2 = inputshape
        inputshape2[0] = inputshape[0]//2
        inputshape2[1] = inputshape[1]//2
        for i in range(3):
            if self._show_sizes:
                print( "coord_t",coord_t[i].shape)
                print( "input_t",input_t[i].shape)
        y = (coord_t[0],input_t[0],batchsize)
        z = (coord_t[1],input_t[1],batchsize)
        w = (coord_t[2],input_t[2],batchsize)
        n = [y,z,w]
        # runs each plane through:
        for i in range(3):
            n[i] = (coord_t[i],input_t[i],batchsize)
            n[i]=self.input(n[i])
            if self._show_sizes:
                print ("inputlayer:",n[i].features.shape)
            n[i]=self.conv1(n[i])
            if self._show_sizes:
                print ("conv1:",n[i].features.shape)
            n[i]=self.maxPool(n[i])
            if self._show_sizes:
                print ("maxPool:",n[i].features.shape)
            for r in range(self._inReps):
                tic = time.perf_counter()
                n[i]=self.SEResNetB2(n[i], inputshape2)
                toc = time.perf_counter()
                print(f"SEResNetB2 in {toc - tic:0.4f} seconds")
                if self._show_sizes:
                    print ("SEResNetB2:",n[i].features.shape)
            n[i]=self.makeDense(n[i])
        # concatenate the inputs while dense, then sparsifies
        if self._show_sizes:
            print ("makeDense:",n[i].shape)
        x = self.concatenateInputs(n)
        if self._show_sizes:
            print ("Concatenate:",x.shape)
        x = self.makeSparse(x)
        if self._show_sizes:
            print("size of sparse after concat:",x.features.shape)
        
        # update inputshape after concatenate
        inputshape2[0] = inputshape2[0]*3
        tic = time.perf_counter()
        # run through the SEResNet Block N (modeled off resnet-34)
        x = self.SEResNetBN(x, inputshape2)
        toc = time.perf_counter()
        print(f"SEResNetBN in {toc - tic:0.4f} seconds")
        if self._show_sizes:
            print ("SEResNetBN:",x.features.shape)
            print("After SEResNetBN x:",x)
        # update inputshape after SEResNetBN
        inputshape2[0] = inputshape2[0]//2 + 2
        inputshape2[1] = inputshape2[1]//2 + 1
        inputshape2[0] = inputshape2[0]//2 + 2
        inputshape2[1] = inputshape2[1]//2 + 2
        inputshape2[0] = inputshape2[0]//2 + 1
        inputshape2[1] = inputshape2[1]//2 + 1
        tic = time.perf_counter()
        x = self.outputAvg(x, inputshape)
        toc = time.perf_counter()
        print(f"outputAvg in {toc - tic:0.4f} seconds")
                
        if self._show_sizes:
            print("before flatten:",x.features.shape)
        x=self.output(x.features)
        
        # output layers
        if self._show_sizes:
            print ("output:",x.shape)
        a = self.flavors(x).view(1,4)
        if self._show_sizes:
            print ("flavors:",a.shape)
        b = self.interactionType(x).view(1,4)
        if self._show_sizes:
            print ("Interaction type:",b.shape)
        c = self.nProton(x).view(1,4)
        if self._show_sizes:
            print ("num protons:",c.shape)
        d = self.nCPion(x).view(1,4)
        if self._show_sizes:
            print ("num charged pions:",d.shape)
        e = self.nNPion(x).view(1,4)
        if self._show_sizes:
            print ("num neutral pions:",e.shape)
        f = self.nNeutron(x).view(1,4)
        if self._show_sizes:
            print ("num Neutrons:",f.shape)
        out = [a,b,c,d,e,f]
        if self._show_sizes:
            print("Final output:",out)
        gtoc = time.perf_counter()
        print(f"Full network in {gtoc - gtic:0.4f} seconds")
        return out
        
    def concatenateInputs(self, input):
        out = torch.cat(input,2)        
        return out
