# Imports
import os,sys,time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# sys.path.append("/home/ebengh01/SparseConvNet")
import sparseconvnet as scn
import se_module as se
sys.path.append("/home/ebengh01/ubdl/Elias_project/networks/models/dense_model")
import dense_se_module as dse

import time
import math
import numpy as np

class SEResNetBN(nn.Module):
    def __init__(self, dimension, in_channels, out_channels):
    
        nn.Module.__init__(self)
        self.nIn = in_channels
        self.nOut = out_channels
        # self.inShape = inputshape
        self.dim = dimension
        self.postRes = scn.ReLU()
        self.postSE = scn.ReLU()
        self.purpleBlockP1 = scn.Sequential(
                               scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                               scn.BatchNormReLU(self.nIn),
                               scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                               scn.BatchNormalization(self.nIn)
                           )
        self.PurpleSE = se.SELayer(self.nIn)
        self.nOut = self.nIn*2
        self.greenResPad = scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 1, False)
        self.greenTransitionBlockP1 = scn.Sequential(
                                        scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                        scn.BatchNormReLU(self.nOut),
                                        scn.SubmanifoldConvolution(self.dim, self.nOut, self.nOut, 3, False),
                                        scn.BatchNormalization(self.nOut)
                                    )
        self.greenTSE = se.SELayer(self.nOut)
        self.nIn = self.nOut
        self.greenBlockP1 = scn.Sequential(
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormReLU(self.nIn),
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormalization(self.nIn)
                            )
        self.greenSE = se.SELayer(self.nIn)
        self.nOut = self.nIn*2
        self.orangeResPad = scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 1, False)
        self.orangeTransitionBlockP1 = scn.Sequential(
                                        scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                        scn.BatchNormReLU(self.nOut),
                                        scn.SubmanifoldConvolution(self.dim, self.nOut, self.nOut, 3, False),
                                        scn.BatchNormalization(self.nOut)
                                    )
        self.orangeTSE = se.SELayer(self.nOut)
        self.nIn = self.nOut
        self.orangeBlockP1 = scn.Sequential(
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormReLU(self.nIn),
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormalization(self.nIn)
                            )
        self.orangeSE = se.SELayer(self.nIn)
        self.nOut = self.nIn*2
        self.blueResPad = scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 1, False)
        self.blueTransitionBlockP1 = scn.Sequential(
                                        scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                        scn.BatchNormReLU(self.nOut),
                                        scn.SubmanifoldConvolution(self.dim, self.nOut, self.nOut, 3, False),
                                        scn.BatchNormalization(self.nOut)
                                    )
        self.blueTSE = se.SELayer(self.nOut)
        self.nIn = self.nOut
        self.blueBlockP1 = scn.Sequential(
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormReLU(self.nIn),
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormalization(self.nIn)
                            )
        self.blueSE = se.SELayer(self.nIn)
    
    
    def forward(self, x, inputshape):
        for i in range(3):
            res = x.features
            tic = time.perf_counter()
            x = self.purpleBlock(x, inputshape)
            toc = time.perf_counter()
            print(f"purpleBlock in {toc - tic:0.4f} seconds")
            x.features += res
            x = self.postRes(x)
            print("postPurple shape:",x.features.shape)
    
        inputshape[0] = inputshape[0]//2 + 2
        inputshape[1] = inputshape[1]//2 + 1
        tic = time.perf_counter()
        res = self.greenResPad(x)
        toc = time.perf_counter()
        print(f"greenResPad in {toc - tic:0.4f} seconds")
        print("post greenResPad shape:",res.features.shape)
        tic = time.perf_counter()
        x = self.greenTransitionBlock(x, inputshape)
        toc = time.perf_counter()
        print(f"greenTransitionBlock in {toc - tic:0.4f} seconds")
        x.features += res.features
        x = self.postRes(x)
        print("green shape: ",x.features.shape)
        for i in range(3):
            res = x.features
            tic = time.perf_counter()
            x = self.greenBlock(x, inputshape)
            toc = time.perf_counter()
            print(f"greenBlock in {toc - tic:0.4f} seconds")
            x.features += res
            x = self.postRes(x)
            print("green shape:",x.features.shape)
    
        inputshape[0] = inputshape[0]//2 + 2
        inputshape[1] = inputshape[1]//2 + 2
        res = self.orangeResPad(x)
        print("post orangeResPad shape:",res.features.shape)
        tic = time.perf_counter()
        x = self.orangeTransitionBlock(x, inputshape)
        toc = time.perf_counter()
        print(f"orangeTransitionBlock in {toc - tic:0.4f} seconds")
        x.features += res.features
        x = self.postRes(x)
        print("orange shape:",x.features.shape)
        for i in range(5):
            res = x.features
            tic = time.perf_counter()
            x = self.orangeBlock(x, inputshape)
            toc = time.perf_counter()
            print(f"orangeBlock in {toc - tic:0.4f} seconds")
            x.features += res
            x = self.postRes(x)
            print("orange shape:",x.features.shape)
    
        inputshape[0] = inputshape[0]//2 + 1
        inputshape[1] = inputshape[1]//2 + 1
        res = self.blueResPad(x)
        tic = time.perf_counter()
        x = self.blueTransitionBlock(x, inputshape)
        toc = time.perf_counter()
        print(f"blueTransitionBlock in {toc - tic:0.4f} seconds")
        x.features += res.features
        x = self.postRes(x)
        print("blue shape:",x.features.shape)
        for i in range(2):
            res = x.features
            tic = time.perf_counter()
            x = self.blueBlock(x, inputshape)
            toc = time.perf_counter()
            print(f"blueBlock in {toc - tic:0.4f} seconds")
            x.features += res
            x = self.postRes(x)
            print("blue shape:",x.features.shape)
        return x
    
    
    def purpleBlock(self, x, inputshape):
        x = self.purpleBlockP1(x)
        x = self.PurpleSE(x, inputshape)
        x = self.postSE(x)
        return x
    
    def greenTransitionBlock(self, x, inputshape):
        x = self.greenTransitionBlockP1(x)
        x = self.greenTSE(x, inputshape)
        x = self.postSE(x)
        return x
    
    def greenBlock(self, x, inputshape):
        x = self.greenBlockP1(x)
        x = self.greenSE(x, inputshape)
        x = self.postSE(x)
        return x
    
    def orangeTransitionBlock(self, x, inputshape):
        x = self.orangeTransitionBlockP1(x)
        x = self.orangeTSE(x, inputshape)
        x = self.postSE(x)
        return x
    
    def orangeBlock(self, x, inputshape):
        x = self.orangeBlockP1(x)
        x = self.orangeSE(x, inputshape)
        x = self.postSE(x)
        return x
    
    def blueTransitionBlock(self, x, inputshape):
        x = self.blueTransitionBlockP1(x)
        x = self.blueTSE(x, inputshape)
        x = self.postSE(x)
        return x
    
    def blueBlock(self, x, inputshape):
        x = self.blueBlockP1(x)
        x = self.blueSE(x, inputshape)
        x = self.postSE(x)
        return x
            
    # SPARSE IMPLEMENTATION
    # def __init__(self, dimension, in_channels, out_channels):
    # 
    #     nn.Module.__init__(self)
    #     self.nIn = in_channels
    #     self.nOut = out_channels
    #     # self.inShape = inputshape
    #     self.dim = dimension
    #     self.postRes = scn.ReLU()
    #     self.postSE = scn.ReLU()
    #     self.purpleBlockP1 = scn.Sequential(
    #                            scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                            scn.BatchNormReLU(self.nIn),
    #                            scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                            scn.BatchNormalization(self.nIn)
    #                        )
    #     self.PurpleSE = se.SELayer(self.nIn)
    #     self.nOut = self.nIn*2
    #     self.greenResPad = scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 1, False)
    #     self.greenTransitionBlockP1 = scn.Sequential(
    #                                     scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                                     scn.BatchNormReLU(self.nOut),
    #                                     scn.SubmanifoldConvolution(self.dim, self.nOut, self.nOut, 3, False),
    #                                     scn.BatchNormalization(self.nOut)
    #                                 )
    #     self.greenTSE = se.SELayer(self.nOut)
    #     self.nIn = self.nOut
    #     self.greenBlockP1 = scn.Sequential(
    #                             scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                             scn.BatchNormReLU(self.nIn),
    #                             scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                             scn.BatchNormalization(self.nIn)
    #                         )
    #     self.greenSE = se.SELayer(self.nIn)
    #     self.nOut = self.nIn*2
    #     self.orangeResPad = scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 1, False)
    #     self.orangeTransitionBlockP1 = scn.Sequential(
    #                                     scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                                     scn.BatchNormReLU(self.nOut),
    #                                     scn.SubmanifoldConvolution(self.dim, self.nOut, self.nOut, 3, False),
    #                                     scn.BatchNormalization(self.nOut)
    #                                 )
    #     self.orangeTSE = se.SELayer(self.nOut)
    #     self.nIn = self.nOut
    #     self.orangeBlockP1 = scn.Sequential(
    #                             scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                             scn.BatchNormReLU(self.nIn),
    #                             scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                             scn.BatchNormalization(self.nIn)
    #                         )
    #     self.orangeSE = se.SELayer(self.nIn)
    #     self.nOut = self.nIn*2
    #     self.blueResPad = scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 1, False)
    #     self.blueTransitionBlockP1 = scn.Sequential(
    #                                     scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                                     scn.BatchNormReLU(self.nOut),
    #                                     scn.SubmanifoldConvolution(self.dim, self.nOut, self.nOut, 3, False),
    #                                     scn.BatchNormalization(self.nOut)
    #                                 )
    #     self.blueTSE = se.SELayer(self.nOut)
    #     self.nIn = self.nOut
    #     self.blueBlockP1 = scn.Sequential(
    #                             scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                             scn.BatchNormReLU(self.nIn),
    #                             scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                             scn.BatchNormalization(self.nIn)
    #                         )
    #     self.blueSE = se.SELayer(self.nIn)
    # 
    # 
    # def forward(self, x, inputshape):
    #     for i in range(3):
    #         res = x.features
    #         tic = time.perf_counter()
    #         x = self.purpleBlock(x, inputshape)
    #         toc = time.perf_counter()
    #         print(f"purpleBlock in {toc - tic:0.4f} seconds")
    #         x.features += res
    #         x = self.postRes(x)
    #         print("postPurple shape:",x.features.shape)
    # 
    #     inputshape[0] = inputshape[0]//2 + 2
    #     inputshape[1] = inputshape[1]//2 + 1
    #     tic = time.perf_counter()
    #     res = self.greenResPad(x)
    #     toc = time.perf_counter()
    #     print(f"greenResPad in {toc - tic:0.4f} seconds")
    #     print("post greenResPad shape:",res.features.shape)
    #     tic = time.perf_counter()
    #     x = self.greenTransitionBlock(x, inputshape)
    #     toc = time.perf_counter()
    #     print(f"greenTransitionBlock in {toc - tic:0.4f} seconds")
    #     x.features += res.features
    #     x = self.postRes(x)
    #     print("green shape: ",x.features.shape)
    #     for i in range(3):
    #         res = x.features
    #         tic = time.perf_counter()
    #         x = self.greenBlock(x, inputshape)
    #         toc = time.perf_counter()
    #         print(f"greenBlock in {toc - tic:0.4f} seconds")
    #         x.features += res
    #         x = self.postRes(x)
    #         print("green shape:",x.features.shape)
    # 
    #     inputshape[0] = inputshape[0]//2 + 2
    #     inputshape[1] = inputshape[1]//2 + 2
    #     res = self.orangeResPad(x)
    #     print("post orangeResPad shape:",res.features.shape)
    #     tic = time.perf_counter()
    #     x = self.orangeTransitionBlock(x, inputshape)
    #     toc = time.perf_counter()
    #     print(f"orangeTransitionBlock in {toc - tic:0.4f} seconds")
    #     x.features += res.features
    #     x = self.postRes(x)
    #     print("orange shape:",x.features.shape)
    #     for i in range(5):
    #         res = x.features
    #         tic = time.perf_counter()
    #         x = self.orangeBlock(x, inputshape)
    #         toc = time.perf_counter()
    #         print(f"orangeBlock in {toc - tic:0.4f} seconds")
    #         x.features += res
    #         x = self.postRes(x)
    #         print("orange shape:",x.features.shape)
    # 
    #     inputshape[0] = inputshape[0]//2 + 1
    #     inputshape[1] = inputshape[1]//2 + 1
    #     res = self.blueResPad(x)
    #     tic = time.perf_counter()
    #     x = self.blueTransitionBlock(x, inputshape)
    #     toc = time.perf_counter()
    #     print(f"blueTransitionBlock in {toc - tic:0.4f} seconds")
    #     x.features += res.features
    #     x = self.postRes(x)
    #     print("blue shape:",x.features.shape)
    #     for i in range(2):
    #         res = x.features
    #         tic = time.perf_counter()
    #         x = self.blueBlock(x, inputshape)
    #         toc = time.perf_counter()
    #         print(f"blueBlock in {toc - tic:0.4f} seconds")
    #         x.features += res
    #         x = self.postRes(x)
    #         print("blue shape:",x.features.shape)
    #     return x
    # 
    # 
    # def purpleBlock(self, x, inputshape):
    #     x = self.purpleBlockP1(x)
    #     x = self.PurpleSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
    # 
    # def greenTransitionBlock(self, x, inputshape):
    #     x = self.greenTransitionBlockP1(x)
    #     x = self.greenTSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
    # 
    # def greenBlock(self, x, inputshape):
    #     x = self.greenBlockP1(x)
    #     x = self.greenSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
    # 
    # def orangeTransitionBlock(self, x, inputshape):
    #     x = self.orangeTransitionBlockP1(x)
    #     x = self.orangeTSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
    # 
    # def orangeBlock(self, x, inputshape):
    #     x = self.orangeBlockP1(x)
    #     x = self.orangeSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
    # 
    # def blueTransitionBlock(self, x, inputshape):
    #     x = self.blueTransitionBlockP1(x)
    #     x = self.blueTSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
    # 
    # def blueBlock(self, x, inputshape):
    #     x = self.blueBlockP1(x)
    #     x = self.blueSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
    
    # HALF DENSE IMPLEMENTATION
    # def __init__(self, dimension, in_channels, out_channels):
    # 
    #     nn.Module.__init__(self)
    #     self.nIn = in_channels
    #     self.nOut = out_channels
    #     # self.inShape = inputshape
    #     self.dim = dimension
    #     self.postRes = scn.ReLU()
    #     self.postSE = scn.ReLU()
    #     self.purpleBlockP1 = scn.Sequential(
    #                            scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                            scn.BatchNormReLU(self.nIn),
    #                            scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                            scn.BatchNormalization(self.nIn)
    #                        )
    #     self.PurpleSE = se.SELayer(self.nIn)
    #     self.nOut = self.nIn*2
    #     self.greenMakeDense = scn.SparseToDense(self.dim, self.nIn)
    #     self.greenResPad = nn.Conv2d(self.nIn, self.nOut, 1, stride=2, padding=1, bias=False)
    #     self.greenTransitionBlock = nn.Sequential(
    #                                     nn.Conv2d(self.nIn, self.nOut, 3, stride=2, padding=2, bias=False),
    #                                     nn.BatchNorm2d(self.nOut, eps=0.0001, momentum=0.99),
    #                                     nn.ReLU(),
    #                                     nn.Conv2d(self.nOut, self.nOut, 3, padding=1, bias=False),
    #                                     nn.BatchNorm2d(self.nOut, eps=0.0001, momentum=0.99),
    #                                     dse.SELayer(self.nOut), # note, may remove batchnorm for optimization
    #                                     nn.ReLU()
    #                                 )
    #     self.greenMakeSparse = scn.DenseToSparse(self.dim)
    #     self.greenTSE = se.SELayer(self.nOut)
    #     self.nIn = self.nOut
    #     self.greenBlockP1 = scn.Sequential(
    #                             scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                             scn.BatchNormReLU(self.nIn),
    #                             scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                             scn.BatchNormalization(self.nIn)
    #                         )
    #     self.greenSE = se.SELayer(self.nIn)
    #     self.nOut = self.nIn*2
    #     self.orangeResPad = scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 1, False)
    #     self.orangeTransitionBlockP1 = scn.Sequential(
    #                                     scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                                     scn.BatchNormReLU(self.nOut),
    #                                     scn.SubmanifoldConvolution(self.dim, self.nOut, self.nOut, 3, False),
    #                                     scn.BatchNormalization(self.nOut)
    #                                 )
    #     self.orangeTSE = se.SELayer(self.nOut)
    #     self.nIn = self.nOut
    #     self.orangeBlockP1 = scn.Sequential(
    #                             scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                             scn.BatchNormReLU(self.nIn),
    #                             scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                             scn.BatchNormalization(self.nIn)
    #                         )
    #     self.orangeSE = se.SELayer(self.nIn)
    #     self.nOut = self.nIn*2
    #     self.blueResPad = scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 1, False)
    #     self.blueTransitionBlockP1 = scn.Sequential(
    #                                     scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                                     scn.BatchNormReLU(self.nOut),
    #                                     scn.SubmanifoldConvolution(self.dim, self.nOut, self.nOut, 3, False),
    #                                     scn.BatchNormalization(self.nOut)
    #                                 )
    #     self.blueTSE = se.SELayer(self.nOut)
    #     self.nIn = self.nOut
    #     self.blueBlockP1 = scn.Sequential(
    #                             scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                             scn.BatchNormReLU(self.nIn),
    #                             scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                             scn.BatchNormalization(self.nIn)
    #                         )
    #     self.blueSE = se.SELayer(self.nIn)
    # 
    # 
    # def forward(self, x, inputshape):
    #     for i in range(3):
    #         res = x.features
    #         tic = time.perf_counter()
    #         x = self.purpleBlock(x, inputshape)
    #         toc = time.perf_counter()
    #         print(f"purpleBlock in {toc - tic:0.4f} seconds")
    #         x.features += res
    #         x = self.postRes(x)
    #         print("postPurple shape:",x.features.shape)
    # 
    #     inputshape[0] = inputshape[0]//2 + 2
    #     inputshape[1] = inputshape[1]//2 + 1
    #     x = self.greenMakeDense(x)
    #     tic = time.perf_counter()
    #     res = self.greenResPad(x)
    #     toc = time.perf_counter()
    #     print(f"greenResPad in {toc - tic:0.4f} seconds")
    #     print("post greenResPad shape:",res.shape)
    #     tic = time.perf_counter()
    #     x = self.greenTransitionBlock(x)
    #     toc = time.perf_counter()
    #     print(f"greenTransitionBlock in {toc - tic:0.4f} seconds")
    #     x += res
    #     x = self.greenMakeSparse(x)
    #     x = self.postRes(x)
    #     print("green shape: ",x.features.shape)
    #     for i in range(3):
    #         res = x.features
    #         tic = time.perf_counter()
    #         x = self.greenBlock(x, inputshape)
    #         toc = time.perf_counter()
    #         print(f"greenBlock in {toc - tic:0.4f} seconds")
    #         x.features += res
    #         x = self.postRes(x)
    #         print("green shape:",x.features.shape)
    # 
    #     inputshape[0] = inputshape[0]//2 + 2
    #     inputshape[1] = inputshape[1]//2 + 2
    #     res = self.orangeResPad(x)
    #     print("post orangeResPad shape:",res.features.shape)
    #     tic = time.perf_counter()
    #     x = self.orangeTransitionBlock(x, inputshape)
    #     toc = time.perf_counter()
    #     print(f"orangeTransitionBlock in {toc - tic:0.4f} seconds")
    #     x.features += res.features
    #     x = self.postRes(x)
    #     print("orange shape:",x.features.shape)
    #     for i in range(5):
    #         res = x.features
    #         tic = time.perf_counter()
    #         x = self.orangeBlock(x, inputshape)
    #         toc = time.perf_counter()
    #         print(f"orangeBlock in {toc - tic:0.4f} seconds")
    #         x.features += res
    #         x = self.postRes(x)
    #         print("orange shape:",x.features.shape)
    # 
    #     inputshape[0] = inputshape[0]//2 + 1
    #     inputshape[1] = inputshape[1]//2 + 1
    #     res = self.blueResPad(x)
    #     tic = time.perf_counter()
    #     x = self.blueTransitionBlock(x, inputshape)
    #     toc = time.perf_counter()
    #     print(f"blueTransitionBlock in {toc - tic:0.4f} seconds")
    #     x.features += res.features
    #     x = self.postRes(x)
    #     print("blue shape:",x.features.shape)
    #     for i in range(2):
    #         res = x.features
    #         tic = time.perf_counter()
    #         x = self.blueBlock(x, inputshape)
    #         toc = time.perf_counter()
    #         print(f"blueBlock in {toc - tic:0.4f} seconds")
    #         x.features += res
    #         x = self.postRes(x)
    #         print("blue shape:",x.features.shape)
    #     return x
    # 
    # 
    # def purpleBlock(self, x, inputshape):
    #     x = self.purpleBlockP1(x)
    #     x = self.PurpleSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
    # 
    # # def greenTransitionBlock(self, x, inputshape):
    # #     x = self.greenTransitionBlockP1(x)
    # #     x = self.greenTSE(x, inputshape)
    # #     x = self.postSE(x)
    # #     return x
    # 
    # def greenBlock(self, x, inputshape):
    #     x = self.greenBlockP1(x)
    #     x = self.greenSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
    # 
    # def orangeTransitionBlock(self, x, inputshape):
    #     x = self.orangeTransitionBlockP1(x)
    #     x = self.orangeTSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
    # 
    # def orangeBlock(self, x, inputshape):
    #     x = self.orangeBlockP1(x)
    #     x = self.orangeSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
    # 
    # def blueTransitionBlock(self, x, inputshape):
    #     x = self.blueTransitionBlockP1(x)
    #     x = self.blueTSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
    # 
    # def blueBlock(self, x, inputshape):
    #     x = self.blueBlockP1(x)
    #     x = self.blueSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x

    # SPARE CONVOLUTION IMPLEMENTATION 
    # def __init__(self, dimension, in_channels, out_channels):
    # 
    #     nn.Module.__init__(self)
    #     self.nIn = in_channels
    #     self.nOut = out_channels
    #     # self.inShape = inputshape
    #     self.dim = dimension
    #     self.postRes = scn.ReLU()
    #     self.postSE = scn.ReLU()
    #     self.purpleBlockP1 = scn.Sequential(
    #                            scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                            scn.BatchNormReLU(self.nIn),
    #                            scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                            scn.BatchNormalization(self.nIn)
    #                        )
    #     self.PurpleSE = se.SELayer(self.nIn)
    #     self.nOut = self.nIn*2
    #     self.greenResPad = scn.Convolution(self.dim, self.nIn, self.nOut, 2, 2, False)
    #     self.greenTransitionBlockP1 = scn.Sequential(
    #                                     scn.Convolution(self.dim, self.nIn, self.nOut, 3, 2, False),
    #                                     scn.BatchNormReLU(self.nOut),
    #                                     scn.SubmanifoldConvolution(self.dim, self.nOut, self.nOut, 3, False),
    #                                     scn.BatchNormalization(self.nOut)
    #                                 )
    #     self.greenTSE = se.SELayer(self.nOut)
    #     self.nIn = self.nOut
    #     self.greenBlockP1 = scn.Sequential(
    #                             scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                             scn.BatchNormReLU(self.nIn),
    #                             scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                             scn.BatchNormalization(self.nIn)
    #                         )
    #     self.greenSE = se.SELayer(self.nIn)
    #     self.nOut = self.nIn*2
    #     self.orangeResPad = scn.Convolution(self.dim, self.nIn, self.nOut, 1, 2, False)
    #     self.orangeTransitionBlockP1 = scn.Sequential(
    #                                     scn.Convolution(self.dim, self.nIn, self.nOut, 3, 2, False),
    #                                     scn.BatchNormReLU(self.nOut),
    #                                     scn.SubmanifoldConvolution(self.dim, self.nOut, self.nOut, 3, False),
    #                                     scn.BatchNormalization(self.nOut)
    #                                 )
    #     self.orangeTSE = se.SELayer(self.nOut)
    #     self.nIn = self.nOut
    #     self.orangeBlockP1 = scn.Sequential(
    #                             scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                             scn.BatchNormReLU(self.nIn),
    #                             scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                             scn.BatchNormalization(self.nIn)
    #                         )
    #     self.orangeSE = se.SELayer(self.nIn)
    #     self.nOut = self.nIn*2
    #     self.blueResPad = scn.Convolution(self.dim, self.nIn, self.nOut, 1, 2, False)
    #     self.blueTransitionBlockP1 = scn.Sequential(
    #                                     scn.Convolution(self.dim, self.nIn, self.nOut, 3, 2, False),
    #                                     scn.BatchNormReLU(self.nOut),
    #                                     scn.SubmanifoldConvolution(self.dim, self.nOut, self.nOut, 3, False),
    #                                     scn.BatchNormalization(self.nOut)
    #                                 )
    #     self.blueTSE = se.SELayer(self.nOut)
    #     self.nIn = self.nOut
    #     self.blueBlockP1 = scn.Sequential(
    #                             scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                             scn.BatchNormReLU(self.nIn),
    #                             scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
    #                             scn.BatchNormalization(self.nIn)
    #                         )
    #     self.blueSE = se.SELayer(self.nIn)
    # 
    # 
    # def forward(self, x, inputshape):
    #     for i in range(3):
    #         res = x.features
    #         tic = time.perf_counter()
    #         x = self.purpleBlock(x, inputshape)
    #         toc = time.perf_counter()
    #         print(f"purpleBlock in {toc - tic:0.4f} seconds")
    #         x.features += res
    #         x = self.postRes(x)
    #         print("postPurple shape:",x.features.shape)
    # 
    #     inputshape[0] = inputshape[0]//2 + 2
    #     inputshape[1] = inputshape[1]//2 + 1
    #     tic = time.perf_counter()
    #     res = self.greenResPad(x)
    #     toc = time.perf_counter()
    #     print(f"greenResPad in {toc - tic:0.4f} seconds")
    #     print("post greenResPad shape:",res.features.shape)
    #     tic = time.perf_counter()
    #     x = self.greenTransitionBlock(x, inputshape)
    #     toc = time.perf_counter()
    #     print(f"greenTransitionBlock in {toc - tic:0.4f} seconds")
    #     x.features += res.features
    #     x = self.postRes(x)
    #     print("green shape: ",x.features.shape)
    #     for i in range(3):
    #         res = x.features
    #         tic = time.perf_counter()
    #         x = self.greenBlock(x, inputshape)
    #         toc = time.perf_counter()
    #         print(f"greenBlock in {toc - tic:0.4f} seconds")
    #         x.features += res
    #         x = self.postRes(x)
    #         print("green shape:",x.features.shape)
    # 
    #     inputshape[0] = inputshape[0]//2 + 2
    #     inputshape[1] = inputshape[1]//2 + 2
    #     res = self.orangeResPad(x)
    #     print("post orangeResPad shape:",res.features.shape)
    #     tic = time.perf_counter()
    #     x = self.orangeTransitionBlock(x, inputshape)
    #     toc = time.perf_counter()
    #     print(f"orangeTransitionBlock in {toc - tic:0.4f} seconds")
    #     x.features += res.features
    #     x = self.postRes(x)
    #     print("orange shape:",x.features.shape)
    #     for i in range(5):
    #         res = x.features
    #         tic = time.perf_counter()
    #         x = self.orangeBlock(x, inputshape)
    #         toc = time.perf_counter()
    #         print(f"orangeBlock in {toc - tic:0.4f} seconds")
    #         x.features += res
    #         x = self.postRes(x)
    #         print("orange shape:",x.features.shape)
    # 
    #     inputshape[0] = inputshape[0]//2 + 1
    #     inputshape[1] = inputshape[1]//2 + 1
    #     res = self.blueResPad(x)
    #     tic = time.perf_counter()
    #     x = self.blueTransitionBlock(x, inputshape)
    #     toc = time.perf_counter()
    #     print(f"blueTransitionBlock in {toc - tic:0.4f} seconds")
    #     x.features += res.features
    #     x = self.postRes(x)
    #     print("blue shape:",x.features.shape)
    #     for i in range(2):
    #         res = x.features
    #         tic = time.perf_counter()
    #         x = self.blueBlock(x, inputshape)
    #         toc = time.perf_counter()
    #         print(f"blueBlock in {toc - tic:0.4f} seconds")
    #         x.features += res
    #         x = self.postRes(x)
    #         print("blue shape:",x.features.shape)
    #     return x
    # 
    # 
    # def purpleBlock(self, x, inputshape):
    #     x = self.purpleBlockP1(x)
    #     x = self.PurpleSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
    # 
    # def greenTransitionBlock(self, x, inputshape):
    #     x = self.greenTransitionBlockP1(x)
    #     x = self.greenTSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
    # 
    # def greenBlock(self, x, inputshape):
    #     x = self.greenBlockP1(x)
    #     x = self.greenSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
    # 
    # def orangeTransitionBlock(self, x, inputshape):
    #     x = self.orangeTransitionBlockP1(x)
    #     x = self.orangeTSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
    # 
    # def orangeBlock(self, x, inputshape):
    #     x = self.orangeBlockP1(x)
    #     x = self.orangeSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
    # 
    # def blueTransitionBlock(self, x, inputshape):
    #     x = self.blueTransitionBlockP1(x)
    #     x = self.blueTSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
    # 
    # def blueBlock(self, x, inputshape):
    #     x = self.blueBlockP1(x)
    #     x = self.blueSE(x, inputshape)
    #     x = self.postSE(x)
    #     return x
