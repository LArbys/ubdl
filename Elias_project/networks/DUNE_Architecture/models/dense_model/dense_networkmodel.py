# Imports
import os,sys,time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# sys.path.append("/home/ebengh01/SparseConvNet")
import sparseconvnet as scn
import dense_se_module as se # can get rid of this eventually
import dense_SEResNetB2 as b2
import dense_SEResNetBN as bn
# import SparseGlobalAvgPool2d as gavg

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
        
        self.SEResNetB2_1 = b2.SEResNetB2(self._dimension, self._nin_features, self._nout_features)
        self.SEResNetB2_2 = b2.SEResNetB2(self._dimension, self._nin_features, self._nout_features)
        self.SEResNetB2_3 = b2.SEResNetB2(self._dimension, self._nin_features, self._nout_features)
        self.makeDense = scn.SparseToDense(self._dimension, self._nin_features)
        # concatenate in forward pass
        self.makeSparse = scn.DenseToSparse(self._dimension)
        inputshape3 = self.inputshape2
        self._nin_features = 3*self._nin_features
        self._nout_features = self._nin_features
        self.SEResNetBN = bn.SEResNetBN(self._dimension, self._nin_features, self._nout_features)
        
        self._nin_features = 16
        self._nout_features = 16
        inputshape4 = [190,218]
        
        self.outputAvg = nn.AdaptiveAvgPool2d(1)
        self.makeDense2 = scn.SparseToDense(self._dimension, self._nout_features)
        self.final_conv = nn.Conv2d(self._nin_features,self._nout_features,(65,217))
        self.output = nn.Sequential(
                          nn.Flatten(),
                          nn.Linear(self._nout_features, 1000)
                      )
        self.flavors = nn.Linear(1000, 3)
        self.interactionType = nn.Linear(1000, 4)
        self.nProton = nn.Linear(1000, 4)
        self.nCPion = nn.Linear(1000, 4)
        self.nNPion = nn.Linear(1000, 4)
        self.nNeutron = nn.Linear(1000, 4)
        self.SoftMax = nn.Softmax(1)
        
    def forward(self,coord_t,input_t,batchsize, device):
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
            print("before input layer:", n[i])
            n[i]=self.input(n[i])
            print("after input layer:",n[i])
            if self._show_sizes:
                print ("inputlayer:",n[i].features.shape)
                print ("inputlayer:",n[i].spatial_size)
            n[i]=self.conv1(n[i])
            if self._show_sizes:
                print ("conv1:",n[i].features.shape)
                print ("conv1:",n[i].spatial_size)
            n[i]=self.maxPool(n[i])
            if self._show_sizes:
                print ("maxPool:",n[i].features.shape)
                print ("maxPool:",n[i].spatial_size)
            tic = time.perf_counter()
            n[i]=self.SEResNetB2_1(n[i], inputshape2, device)
            n[i]=self.SEResNetB2_2(n[i], inputshape2, device)
            n[i]=self.SEResNetB2_3(n[i], inputshape2, device)
            toc = time.perf_counter()
            print(f"SEResNetB2 in {toc - tic:0.4f} seconds")
            if self._show_sizes:
                print ("SEResNetB2:",n[i].features.shape)
                print ("SEResNetB2:",n[i].spatial_size)
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
            print("size of sparse after concat:",x.spatial_size)
        
        tic = time.perf_counter()
        # run through the SEResNet Block N (modeled off resnet-34)
        x = self.SEResNetBN(x, inputshape2, device)
        toc = time.perf_counter()
        print(f"SEResNetBN in {toc - tic:0.4f} seconds")
        if self._show_sizes:
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
        x = self.final_conv(x)
        if self._show_sizes:
            print("before flatten:",x.shape)
        x=self.output(x)
        # output layers
        if self._show_sizes:
            print ("output:",x.shape)
        a = self.flavors(x).view(1,3)
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
        
    def deploy(self,coord_t,input_t,batchsize,device):
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
                print ("inputlayer:",n[i].spatial_size)
            n[i]=self.conv1(n[i])
            if self._show_sizes:
                print ("conv1:",n[i].features.shape)
                print ("conv1:",n[i].spatial_size)
            n[i]=self.maxPool(n[i])
            if self._show_sizes:
                print ("maxPool:",n[i].features.shape)
                print ("maxPool:",n[i].spatial_size)
            tic = time.perf_counter()
            n[i]=self.SEResNetB2_1(n[i], inputshape2, device)
            n[i]=self.SEResNetB2_2(n[i], inputshape2, device)
            n[i]=self.SEResNetB2_3(n[i], inputshape2, device)
            toc = time.perf_counter()
            print(f"SEResNetB2 in {toc - tic:0.4f} seconds")
            if self._show_sizes:
                print ("SEResNetB2:",n[i].features.shape)
                print ("SEResNetB2:",n[i].spatial_size)
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
            print("size of sparse after concat:",x.spatial_size)
        
        tic = time.perf_counter()
        # run through the SEResNet Block N (modeled off resnet-34)
        x = self.SEResNetBN(x, inputshape2, device)
        toc = time.perf_counter()
        print(f"SEResNetBN in {toc - tic:0.4f} seconds")
        if self._show_sizes:
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
        x = self.final_conv(x)
        if self._show_sizes:
            print("before flatten:",x.shape)
        x=self.output(x)
        # output layers
        if self._show_sizes:
            print ("output:",x.shape)
        a = self.flavors(x).view(1,3)
        a = self.SoftMax(a)
        if self._show_sizes:
            print ("flavors:",a.shape)
        b = self.interactionType(x).view(1,4)
        b = self.SoftMax(b)
        if self._show_sizes:
            print ("Interaction type:",b.shape)
        c = self.nProton(x).view(1,4)
        c = self.SoftMax(c)
        if self._show_sizes:
            print ("num protons:",c.shape)
        d = self.nCPion(x).view(1,4)
        d = self.SoftMax(d)
        if self._show_sizes:
            print ("num charged pions:",d.shape)
        e = self.nNPion(x).view(1,4)
        e = self.SoftMax(e)
        if self._show_sizes:
            print ("num neutral pions:",e.shape)
        f = self.nNeutron(x).view(1,4)
        f = self.SoftMax(f)
        if self._show_sizes:
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



# # Imports
# import os,sys,time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# 
# # sys.path.append("/home/ebengh01/SparseConvNet")
# import sparseconvnet as scn
# import dense_se_module as se # can get rid of this eventually
# import dense_SEResNetB2 as b2
# import dense_SEResNetBN as bn
# 
# import time
# import math
# import numpy as np
# 
# class SparseClassifier(nn.Module):
#     def __init__(self, inputshape, nin_features, nout_features, nplanes,show_sizes):
#         nn.Module.__init__(self)
#         """
#         inputs
#         ------
#         inputshape [list of int]: dimensions of the matrix or image
#         nin_features [int]: number of features in the first convolutional layer
#         nout_features [int]: number of features that feed into the regression layer
#         nPlanes [int]: the depth of the U-Net
#         show_sizes [bool]: if True, print sizes while running forward
#         """
#         self._mode = 0
#         self._dimension = 2
#         self._inputshape = inputshape
#         if len(self._inputshape)!=self._dimension:
#             raise ValueError("expected inputshape to contain size of 2 dimensions only."
#                              +"given %d values"%(len(self._inputshape)))
#         self._inReps = 3
#         self._nin_features = nin_features
#         self._nout_features = nout_features
#         self._nplanes = [nin_features, 2*nin_features, 3*nin_features, 4*nin_features, 5*nin_features]
#         self._show_sizes = show_sizes        
# 
#         self.input = scn.InputLayer(self._dimension, self._inputshape, mode=self._mode)
#         self.conv1 = scn.SubmanifoldConvolution(self._dimension, 1, self._nin_features, 7, False)
#         self.maxPool = scn.MaxPooling(self._dimension,2,2)
#         self.makeDense = scn.SparseToDense(self._dimension, self._nin_features)
#         self.SEResNetB2 = b2.SEResNetB2(self._nin_features, self._nout_features, self._inReps)
#         # concatenate in forward pass
#         self.SEResNetBN = bn.SEResNetBN(self._nin_features, self._nout_features)
#         self._nin_features = self._nin_features*8
#         self._nout_features = self._nout_features*8
#         self.output = nn.Sequential(
#                           nn.AdaptiveAvgPool2d(1),
#                           nn.Flatten(),
#                           nn.Linear(512, 1000)
#                       )
#         self.flavors = nn.Sequential(
#                       nn.Linear(1000, 4),
#                       nn.Softmax(1)
#                   )
#         self.interactionType = nn.Sequential(
#                       nn.Linear(1000, 4),
#                       nn.Softmax(1)
#                   )
#         self.nProton = nn.Sequential(
#                       nn.Linear(1000, 4),
#                       nn.Softmax(1)
#                   )
#         self.nCPion = nn.Sequential(
#                       nn.Linear(1000, 4),
#                       nn.Softmax(1)
#                   )
#         self.nNPion = nn.Sequential(
#                       nn.Linear(1000, 4),
#                       nn.Softmax(1)
#                   )
#         self.nNeutron = nn.Sequential(
#                       nn.Linear(1000, 4),
#                       nn.Softmax(1)
#                   )
# 
#     def forward(self,coord_t,input_t,batchsize):
#         gtic = time.perf_counter()
#         for i in range(3):
#             print("Input: ", i)
#             if self._show_sizes:
#                 print( "coord_t",coord_t[i].shape)
#                 print( "input_t",input_t[i].shape)
#         y = (coord_t[0],input_t[0],batchsize)
#         z = (coord_t[1],input_t[1],batchsize)
#         w = (coord_t[2],input_t[2],batchsize)
#         n = [y,z,w]
#         for i in range(3):
#             print("Input:", i)
#             n[i] = (coord_t[i],input_t[i],batchsize)
#             n[i]=self.input(n[i])
#             if self._show_sizes:
#                 print ("inputlayer:",n[i].features.shape)
#             n[i]=self.conv1(n[i])
#             if self._show_sizes:
#                 print ("conv1:",n[i].features.shape)
#             n[i]=self.maxPool(n[i])
#             if self._show_sizes:
#                 print ("maxPool:",n[i].features.shape)
#                 # print("data after maxPool:",n[i])
#             n[i]=self.makeDense(n[i])
#             # print("before output:",n[i].shape)
#             # n[i] = self.output(n[i])
#             # print("after output:",n[i].shape)
# 
#             if self._show_sizes:
#                 print ("makeDense:",n[i].shape)
#             for r in range(self._inReps):
#                 n[i]=self.SEResNetB2(n[i])
#                 if self._show_sizes:
#                     print ("SEResNetB2:",n[i].shape)
#         x = self.concatenateInputs(n)
#         if self._show_sizes:
#             print ("Concatenate:",x.shape)
#         tic = time.perf_counter()
#         x = self.SEResNetBN(x)
#         toc = time.perf_counter()
#         print(f"SEResNetBN in {toc - tic:0.4f} seconds")
#         if self._show_sizes:
#             print ("SEResNetBN:",x.shape)
#         x=self.output(x)
#         if self._show_sizes:
#             print ("output:",x.shape)
#         a = self.flavors(x)
#         if self._show_sizes:
#             print ("flavors:",a.shape)
#         b = self.interactionType(x)
#         if self._show_sizes:
#             print ("Interaction type:",b.shape)
#         c = self.nProton(x)
#         if self._show_sizes:
#             print ("num protons:",c.shape)
#         d = self.nCPion(x)
#         if self._show_sizes:
#             print ("num charged pions:",d.shape)
#         e = self.nNPion(x)
#         if self._show_sizes:
#             print ("num neutral pions:",e.shape)
#         f = self.nNeutron(x)
#         if self._show_sizes:
#             print ("num Neutrons:",f.shape)
#         out = [a,b,c,d,e,f]
#         print("Final output:",out)
#         gtoc = time.perf_counter()
#         print(f"Full network in {gtoc - gtic:0.4f} seconds")
#         return out
# 
#     def concatenateInputs(self, inputs):
#         out = torch.cat(inputs,2)
#         return out    
# 