# Imports
import os,sys,time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# deprecated/sparse_larflow/utils_sparselarflow.py
sys.path.append("/home/ebengh01/ubdl/larflow/deprecated/sparse_larflow/")
import sparseconvnet as scn
import se_module as se # can get rid of this eventually
import utils_sparselarflow as b2
# import ResNetB2 as b2
import ResNetBN as bn
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
        self.ResNetB2 = b2.create_resnet_layer(3, self._nin_features, self._nout_features)
        self.makeDense = scn.SparseToDense(self._dimension, self._nin_features)
        self.dropout2 = nn.Dropout(p=0.2)
        # concatenate in forward pass
        self.makeSparse = scn.DenseToSparse(self._dimension)
        inputshape3 = self.inputshape2
        self._nin_features = 3*self._nin_features
        self._nout_features = self._nin_features
        self.SEResNetBN = bn.SEResNetBN(self._dimension, self._nin_features, self._nout_features)
        
        
        self._nin_features = 16
        self._nout_features = 16
        inputshape4 = [190,218]
        
        self.outputAvg = gavg.SparseGlobalAvgPool2d()
        self.makeDense2 = scn.SparseToDense(self._dimension, self._nout_features)
        self.final_conv = nn.Conv2d(self._nin_features,self._nout_features,(512, 1728))#513, 1729
        self.final_bnorm = nn.BatchNorm2d(16)
        self.dropout5 = nn.Dropout()
        self.maxpool2 = nn.MaxPool2d(4,padding=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self._nout_features, 16)
        self.output = nn.Sequential(
                          nn.Flatten(),
                          nn.Linear(self._nout_features, 16)
                      )
        self.flavors = nn.Linear(16, 3)
        self.interactionType = nn.Linear(16, 4)
        self.nProton = nn.Linear(16, 4)
        self.nCPion = nn.Linear(16, 4)
        self.nNPion = nn.Linear(16, 4)
        self.nNeutron = nn.Linear(16, 4)
        self.SoftMax = nn.Softmax(1)
        
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
        nan_check_1 = [False, False, False]
        # runs each plane through:
        for i in range(3):
            n[i] = (coord_t[i],input_t[i],batchsize)
            n[i]=self.input(n[i])
            if self._show_sizes:
                print ("inputlayer:",n[i].features.device)
                print ("inputlayer:",n[i].features.shape)
                print ("inputlayer:",n[i].spatial_size)
                nan_check_1[i] = torch.any(torch.isnan(n[i].features))
                if nan_check_1[i]:
                    print("PROBLEM 1:",n[i])
            n[i]=self.conv1(n[i])
            if self._show_sizes:
                print ("conv1:",n[i].features.device)
                print ("conv1:",n[i].features.shape)
                print ("conv1:",n[i].spatial_size)
            n[i]=self.maxPool(n[i])
            if self._show_sizes:
                print ("maxPool:",n[i].features.device)
                print ("maxPool:",n[i].features.shape)
                print ("maxPool:",n[i].spatial_size)
            tic = time.perf_counter()
            n[i]=self.ResNetB2(n[i])
            toc = time.perf_counter()
            print(f"SEResNetB2 in {toc - tic:0.4f} seconds")
            if self._show_sizes:
                print ("SEResNetB2:",n[i].features.device)
                print ("SEResNetB2:",n[i].features.shape)
                print ("SEResNetB2:",n[i].spatial_size)
            tic = time.perf_counter()
            n[i]=self.makeDense(n[i])
            toc = time.perf_counter()
            print(f"makeDense_%d in {toc - tic:0.4f} seconds"%(i))
            if self._show_sizes:
                print ("makeDense:",n[i].device)
                print ("makeDense:",n[i].shape)
        # concatenate the inputs while dense, then sparsifies
        tic = time.perf_counter()
        x = self.concatenateInputs(n)
        if self._show_sizes:
            print ("Concatenate:",x.device)
            print ("Concatenate:",x.shape)
            nan_check_2 = torch.any(torch.isnan(x))
            if nan_check_2:
                print("PROBLEM 2:",x)
        x = self.dropout2(x)
        if self._show_sizes:
            print ("dropout2:",x.device)
            print ("dropout2:",x.shape)
        x = self.makeSparse(x)
        if self._show_sizes:
            print("sparse after concat:",x.features.device)
            print("shape of sparse after concat:",x.features.shape)
            print("spatial size of sparse after concat:",x.spatial_size)
        toc = time.perf_counter()
        print(f"Concat and makeSparse in {toc - tic:0.4f} seconds")
        tic = time.perf_counter()
        # run through the SEResNet Block N (modeled off resnet-34)
        x = self.SEResNetBN(x, inputshape2)
        toc = time.perf_counter()
        print(f"SEResNetBN in {toc - tic:0.4f} seconds")
        if self._show_sizes:
            print ("SEResNetBN:",x.features.device)
            print ("SEResNetBN:",x.features.shape)
            print("SEResNetBN:",x.spatial_size)
        # update inputshape after SEResNetBN
        inputshape2[0] = inputshape2[0]//2 + 2
        inputshape2[1] = inputshape2[1]//2 + 1
        inputshape2[0] = inputshape2[0]//2 + 2
        inputshape2[1] = inputshape2[1]//2 + 2
        inputshape2[0] = inputshape2[0]//2 + 1
        inputshape2[1] = inputshape2[1]//2 + 1

        x = self.makeDense2(x)
        if self._show_sizes:
            print("After dense:",x.device)
            print("After dense:",x.shape)
            nan_check_3 = torch.any(torch.isnan(x))
            if nan_check_3:
                print("PROBLEM 3:",x)
        x = self.final_conv(x)
        if self._show_sizes:
            print("after final conv:",x.device)
            print("after final conv:",x.shape)
            nan_check_4 = torch.any(torch.isnan(x))
            if nan_check_4:
                print("PROBLEM 4:",x)
        x = self.final_bnorm(x)
        if self._show_sizes:
            print("after final batchnorm:",x.device)
            print("after final batchnorm:",x.shape)
        x = self.dropout5(x)
        if self._show_sizes:
            print("after dropout5:",x.device)
            print("after dropout5:",x.shape)
        x = self.maxpool2(x)
        if self._show_sizes:
            print("after maxpool2:",x.device)
            print("after maxpool2:",x.shape)
        x = self.flatten(x)
        if self._show_sizes:
            print("after flatten:",x.device)
            print("after flatten:",x.shape)
        x = self.linear(x)
        if self._show_sizes:
            print("after linear:",x.device)
            print("after linear:",x.shape)
        # x=self.output(x)
        if self._show_sizes:
            print("after output:",x.device)
            print("after output:",x.shape)
        # output layers
        a = self.flavors(x).view(1,3)
        a = self.SoftMax(a)
        if self._show_sizes:
            print ("flavors:",a.device)
            print ("flavors:",a.shape)
        b = self.interactionType(x).view(1,4)
        b = self.SoftMax(b)
        if self._show_sizes:
            print ("Interaction type:",b.device)
            print ("Interaction type:",b.shape)
        c = self.nProton(x).view(1,4)
        c = self.SoftMax(c)
        if self._show_sizes:
            print ("num protons:",c.device)
            print ("num protons:",c.shape)
        d = self.nCPion(x).view(1,4)
        d = self.SoftMax(d)
        if self._show_sizes:
            print ("num charged pions:",d.device)
            print ("num charged pions:",d.shape)
        e = self.nNPion(x).view(1,4)
        e = self.SoftMax(e)
        if self._show_sizes:
            print ("num neutral pions:",e.device)
            print ("num neutral pions:",e.shape)
        f = self.nNeutron(x).view(1,4)
        f = self.SoftMax(f)
        if self._show_sizes:
            print ("num Neutrons:",f.device)
            print ("num Neutrons:",f.shape)
        out = [a,b,c,d,e,f]
        if self._show_sizes:
            print("Final output:",out)
        gtoc = time.perf_counter()
        print(f"Full network in {gtoc - gtic:0.4f} seconds")
        return out
        
    def deploy(self,coord_t,input_t,batchsize):
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
        nan_check_1 = [False, False, False]
        # runs each plane through:
        for i in range(3):
            n[i] = (coord_t[i],input_t[i],batchsize)
            n[i]=self.input(n[i])            
            if self._show_sizes:
                print ("inputlayer:",n[i].features.device)
                print ("inputlayer:",n[i].features.shape)
                print ("inputlayer:",n[i].spatial_size)
                nan_check_1[i] = torch.any(torch.isnan(n[i].features))
                if nan_check_1[i]:
                    print("PROBLEM 1, i =:",i)
                    print("n[i]:",n[i])
            n[i]=self.conv1(n[i])
            if self._show_sizes:
                print ("conv1:",n[i].features.device)
                print ("conv1:",n[i].features.shape)
                print ("conv1:",n[i].spatial_size)
            n[i]=self.maxPool(n[i])
            if self._show_sizes:
                print ("maxPool:",n[i].features.device)
                print ("maxPool:",n[i].features.shape)
                print ("maxPool:",n[i].spatial_size)
            tic = time.perf_counter()
            n[i]=self.ResNetB2(n[i])
            toc = time.perf_counter()
            print(f"SEResNetB2 in {toc - tic:0.4f} seconds")
            if self._show_sizes:
                print ("SEResNetB2:",n[i].features.device)
                print ("SEResNetB2:",n[i].features.shape)
                print ("SEResNetB2:",n[i].spatial_size)
                nan_check_1[i] = torch.any(torch.isnan(n[i].features))
                if nan_check_1[i]:
                    print("PROBLEM 2, i =:",i)
                    print("n[i]:",n[i])
            tic = time.perf_counter()
            n[i]=self.makeDense(n[i])
            toc = time.perf_counter()
            print(f"makeDense_%d in {toc - tic:0.4f} seconds"%(i))
            if self._show_sizes:
                print ("makeDense:",n[i].device)
                print ("makeDense:",n[i].shape)
        # concatenate the inputs while dense, then sparsifies
        tic = time.perf_counter()
        x = self.concatenateInputs(n)
        if self._show_sizes:
            print ("Concatenate:",x.device)
            print ("Concatenate:",x.shape)
            nan_check_2 = torch.any(torch.isnan(x))
            if nan_check_2:
                print("PROBLEM 3:",x)
        x = self.makeSparse(x)
        if self._show_sizes:
            print("sparse after concat:",x.features.device)
            print("shape of sparse after concat:",x.features.shape)
            print("spatial size of sparse after concat:",x.spatial_size)
        toc = time.perf_counter()
        print(f"Concat and makeSparse in {toc - tic:0.4f} seconds")
        tic = time.perf_counter()
        # run through the SEResNet Block N (modeled off resnet-34)
        x = self.SEResNetBN(x, inputshape2)
        toc = time.perf_counter()
        print(f"SEResNetBN in {toc - tic:0.4f} seconds")
        if self._show_sizes:
            print ("SEResNetBN:",x.features.device)
            print ("SEResNetBN:",x.features.shape)
            print("SEResNetBN:",x.spatial_size)
        # update inputshape after SEResNetBN
        inputshape2[0] = inputshape2[0]//2 + 2
        inputshape2[1] = inputshape2[1]//2 + 1
        inputshape2[0] = inputshape2[0]//2 + 2
        inputshape2[1] = inputshape2[1]//2 + 2
        inputshape2[0] = inputshape2[0]//2 + 1
        inputshape2[1] = inputshape2[1]//2 + 1

        x = self.makeDense2(x)
        if self._show_sizes:
            print("After dense:",x.device)
            print("After dense:",x.shape)
            nan_check_3 = torch.any(torch.isnan(x))
            if nan_check_3:
                print("PROBLEM 4:",x)
        x = self.final_conv(x)
        if self._show_sizes:
            print("after final conv:",x.device)
            print("after final conv:",x.shape)
            nan_check_4 = torch.any(torch.isnan(x))
            if nan_check_4:
                print("PROBLEM 5:",x)
        x = self.final_bnorm(x)
        if self._show_sizes:
            print("after final batchnorm:",x.device)
            print("after final batchnorm:",x.shape)
        x = self.maxpool2(x)
        if self._show_sizes:
            print("after maxpool2:",x.device)
            print("after maxpool2:",x.shape)
        x = self.flatten(x)
        if self._show_sizes:
            print("after flatten:",x.device)
            print("after flatten:",x.shape)
        x = self.linear(x)
        if self._show_sizes:
            print("after linear:",x.device)
            print("after linear:",x.shape)
        # x=self.output(x)
        if self._show_sizes:
            print("after output:",x.device)
            print("after output:",x.shape)
        # output layers
        a = self.flavors(x).view(1,3)
        a = self.SoftMax(a)
        if self._show_sizes:
            print ("flavors:",a.device)
            print ("flavors:",a.shape)
        b = self.interactionType(x).view(1,4)
        b = self.SoftMax(b)
        if self._show_sizes:
            print ("Interaction type:",b.device)
            print ("Interaction type:",b.shape)
        c = self.nProton(x).view(1,4)
        c = self.SoftMax(c)
        if self._show_sizes:
            print ("num protons:",c.device)
            print ("num protons:",c.shape)
        d = self.nCPion(x).view(1,4)
        d = self.SoftMax(d)
        if self._show_sizes:
            print ("num charged pions:",d.device)
            print ("num charged pions:",d.shape)
        e = self.nNPion(x).view(1,4)
        e = self.SoftMax(e)
        if self._show_sizes:
            print ("num neutral pions:",e.device)
            print ("num neutral pions:",e.shape)
        f = self.nNeutron(x).view(1,4)
        f = self.SoftMax(f)
        if self._show_sizes:
            print ("num Neutrons:",f.device)
            print ("num Neutrons:",f.shape)
        out = [a,b,c,d,e,f]
        if self._show_sizes:
            print("Final output:",out)
        gtoc = time.perf_counter()
        print(f"Full network in {gtoc - gtic:0.4f} seconds")
        return out
    
    def concatenateInputs(self, input):
        out = torch.cat(input,1)        
        return out
