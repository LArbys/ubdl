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
        self.dim = dimension
        self.postRes = scn.LeakyReLU()
        self.postSE = scn.LeakyReLU()
        self.purpleBlockP1_1 = scn.Sequential(
                               scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                               scn.BatchNormLeakyReLU(self.nIn),
                               scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                               scn.BatchNormalization(self.nIn)
                           )
        self.PurpleSE_1 = se.SELayer(self.nIn)
        # self.purpleBlockP1_2 = scn.Sequential(
        #                        scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
        #                        scn.BatchNormLeakyReLU(self.nIn),
        #                        scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
        #                        scn.BatchNormalization(self.nIn)
        #                    )
        # self.PurpleSE_2 = se.SELayer(self.nIn)
        # self.purpleBlockP1_3 = scn.Sequential(
        #                        scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
        #                        scn.BatchNormLeakyReLU(self.nIn),
        #                        scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
        #                        scn.BatchNormalization(self.nIn)
        #                    )
        # self.PurpleSE_3 = se.SELayer(self.nIn)




        self.nOut = self.nIn//3
        self.greenResPad = scn.Sequential(
                                        scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 1, False),
                                        scn.MaxPooling(self.dim, 1, 2)
                                    )
        self.greenTransitionBlockP1 = scn.Sequential(
                                        scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                        scn.MaxPooling(self.dim, 1, 2),
                                        scn.BatchNormLeakyReLU(self.nOut),
                                        scn.SubmanifoldConvolution(self.dim, self.nOut, self.nOut, 3, False),
                                        scn.BatchNormalization(self.nOut)
                                    )
        self.greenTSE = se.SELayer(self.nOut)
        self.nIn = self.nOut
        self.greenBlockP1_1 = scn.Sequential(
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormLeakyReLU(self.nIn),
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormalization(self.nIn)
                            )
        self.greenSE_1 = se.SELayer(self.nIn)
        self.greenBlockP1_2 = scn.Sequential(
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormLeakyReLU(self.nIn),
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormalization(self.nIn)
                            )
        self.greenSE_2 = se.SELayer(self.nIn)
        self.greenBlockP1_3 = scn.Sequential(
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormLeakyReLU(self.nIn),
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormalization(self.nIn)
                            )
        self.greenSE_3 = se.SELayer(self.nIn)
        self.nOut = self.nIn//2
        self.orangeResPad = scn.Sequential(
                                        scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 1, False),
                                        scn.MaxPooling(self.dim, 1, 2)
                                    )
        self.orangeTransitionBlockP1 = scn.Sequential(
                                        scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                        scn.MaxPooling(self.dim, 1, 2),
                                        scn.BatchNormLeakyReLU(self.nOut),
                                        scn.SubmanifoldConvolution(self.dim, self.nOut, self.nOut, 3, False),
                                        scn.BatchNormalization(self.nOut)
                                    )
        self.orangeTSE = se.SELayer(self.nOut)
        self.nIn = self.nOut
        self.orangeBlockP1_1 = scn.Sequential(
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormLeakyReLU(self.nIn),
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormalization(self.nIn)
                            )
        self.orangeSE_1 = se.SELayer(self.nIn)
        self.orangeBlockP1_2 = scn.Sequential(
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormLeakyReLU(self.nIn),
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormalization(self.nIn)
                            )
        self.orangeSE_2 = se.SELayer(self.nIn)
        self.orangeBlockP1_3 = scn.Sequential(
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormLeakyReLU(self.nIn),
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormalization(self.nIn)
                            )
        self.orangeSE_3 = se.SELayer(self.nIn)
        self.orangeBlockP1_4 = scn.Sequential(
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormLeakyReLU(self.nIn),
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormalization(self.nIn)
                            )
        self.orangeSE_4 = se.SELayer(self.nIn)
        self.orangeBlockP1_5 = scn.Sequential(
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormLeakyReLU(self.nIn),
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormalization(self.nIn)
                            )
        self.orangeSE_5 = se.SELayer(self.nIn)

        self.nOut = self.nIn//2
        self.blueResPad = scn.Sequential(
                                        scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 1, False),
                                        scn.MaxPooling(self.dim, 1, 2)
                                    )
        self.blueTransitionBlockP1 = scn.Sequential(
                                        scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                        scn.MaxPooling(self.dim, 1, 2),
                                        scn.BatchNormLeakyReLU(self.nOut),
                                        scn.SubmanifoldConvolution(self.dim, self.nOut, self.nOut, 3, False),
                                        scn.BatchNormalization(self.nOut)
                                    )        
        self.blueTSE = se.SELayer(self.nOut)
        self.nIn = self.nOut
        self.blueBlockP1_1 = scn.Sequential(
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormLeakyReLU(self.nIn),
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormalization(self.nIn)
                            )
        self.blueSE_1 = se.SELayer(self.nIn)
        self.blueBlockP1_2 = scn.Sequential(
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormLeakyReLU(self.nIn),
                                scn.SubmanifoldConvolution(self.dim, self.nIn, self.nOut, 3, False),
                                scn.BatchNormalization(self.nIn)
                            )
        self.blueSE_2 = se.SELayer(self.nIn)
        
    
    def forward(self, x, inputshape, device):
        # Purple Blocks:
        tic = time.perf_counter()
        res = x.features
        x = self.purpleBlockP1_1(x)
        x = self.PurpleSE_1(x, inputshape, device)
        x = self.postSE(x)
        x.features += res
        x = self.postRes(x)
        # res = x.features
        # x = self.purpleBlockP1_2(x)
        # x = self.PurpleSE_2(x, inputshape, device)
        # x = self.postSE(x)
        # x.features += res
        # x = self.postRes(x)
        # res = x.features
        # x = self.purpleBlockP1_3(x)
        # x = self.PurpleSE_3(x, inputshape, device)
        # x = self.postSE(x)
        # x.features += res
        # x = self.postRes(x)
        toc = time.perf_counter()
        print(f"purpleBlock in {toc - tic:0.4f} seconds")
        
        inputshape[0] = inputshape[0]//2 + 1
        inputshape[1] = inputshape[1]//2 + 1
        tic = time.perf_counter()
        res = self.greenResPad(x)
        toc = time.perf_counter()
        print(f"greenResPad in {toc - tic:0.4f} seconds")
        tic = time.perf_counter()
        x = self.greenTransitionBlock(x, inputshape, device)
        toc = time.perf_counter()
        print(f"greenTransitionBlock in {toc - tic:0.4f} seconds")
        x.features += res.features
        x = self.postRes(x)
        tic = time.perf_counter()
        res = x.features
        x = self.greenBlockP1_1(x)
        x = self.greenSE_1(x, inputshape, device)
        x = self.postSE(x)
        x.features += res
        x = self.postRes(x)
        res = x.features
        x = self.greenBlockP1_2(x)
        x = self.greenSE_2(x, inputshape, device)
        x = self.postSE(x)
        x.features += res
        x = self.postRes(x)
        res = x.features
        x = self.greenBlockP1_3(x)
        x = self.greenSE_3(x, inputshape, device)
        x = self.postSE(x)
        x.features += res
        x = self.postRes(x)
        toc = time.perf_counter()
        print(f"greenBlock in {toc - tic:0.4f} seconds")

        inputshape[0] = inputshape[0]//2 + 1
        inputshape[1] = inputshape[1]//2 + 1
        res = self.orangeResPad(x)
        tic = time.perf_counter()
        x = self.orangeTransitionBlock(x, inputshape, device)
        toc = time.perf_counter()
        print(f"orangeTransitionBlock in {toc - tic:0.4f} seconds")
        x.features += res.features
        x = self.postRes(x)
        tic = time.perf_counter()
        res = x.features
        x = self.orangeBlockP1_1(x)
        x = self.orangeSE_1(x, inputshape, device)
        x = self.postSE(x)
        x.features += res
        x = self.postRes(x)
        res = x.features
        x = self.orangeBlockP1_2(x)
        x = self.orangeSE_2(x, inputshape, device)
        x = self.postSE(x)
        x.features += res
        x = self.postRes(x)
        res = x.features
        x = self.orangeBlockP1_3(x)
        x = self.orangeSE_3(x, inputshape, device)
        x = self.postSE(x)
        x.features += res
        x = self.postRes(x)
        res = x.features
        x = self.orangeBlockP1_4(x)
        x = self.orangeSE_4(x, inputshape, device)
        x = self.postSE(x)
        x.features += res
        x = self.postRes(x)
        res = x.features
        x = self.orangeBlockP1_5(x)
        x = self.orangeSE_5(x, inputshape, device)
        x = self.postSE(x)
        x.features += res
        x = self.postRes(x)
        toc = time.perf_counter()
        print(f"orangeBlock in {toc - tic:0.4f} seconds")

        inputshape[0] = inputshape[0]//2 + 1
        inputshape[1] = inputshape[1]//2 + 1
        res = self.blueResPad(x)
        tic = time.perf_counter()
        x = self.blueTransitionBlock(x, inputshape, device)
        toc = time.perf_counter()
        print(f"blueTransitionBlock in {toc - tic:0.4f} seconds")
        x.features += res.features
        x = self.postRes(x)
        tic = time.perf_counter()
        res = x.features
        x = self.blueBlockP1_1(x)
        x = self.blueSE_1(x, inputshape, device)
        x = self.postSE(x)
        x.features += res
        x = self.postRes(x)
        res = x.features
        x = self.blueBlockP1_2(x)
        x = self.blueSE_2(x, inputshape, device)
        x = self.postSE(x)
        x.features += res
        x = self.postRes(x)
        toc = time.perf_counter()
        print(f"blueBlock in {toc - tic:0.4f} seconds")

        return x
    
    
    # def purpleBlock(self, x, inputshape, device):
    #     x = self.purpleBlockP1(x)
    #     x = self.PurpleSE(x, inputshape, device)
    #     x = self.postSE(x)
    #     return x
    
    def greenTransitionBlock(self, x, inputshape, device):
        x = self.greenTransitionBlockP1(x)
        x = self.greenTSE(x, inputshape, device)
        x = self.postSE(x)
        return x
    
    # def greenBlock(self, x, inputshape, device):
    #     x = self.greenBlockP1(x)
    #     x = self.greenSE(x, inputshape, device)
    #     x = self.postSE(x)
    #     return x
    
    def orangeTransitionBlock(self, x, inputshape, device):
        x = self.orangeTransitionBlockP1(x)
        x = self.orangeTSE(x, inputshape, device)
        x = self.postSE(x)
        return x
    
    # def orangeBlock(self, x, inputshape, device):
    #     x = self.orangeBlockP1(x)
    #     x = self.orangeSE(x, inputshape, device)
    #     x = self.postSE(x)
    #     return x
    
    def blueTransitionBlock(self, x, inputshape, device):
        x = self.blueTransitionBlockP1(x)
        x = self.blueTSE(x, inputshape, device)
        x = self.postSE(x)
        return x
    
    # def blueBlock(self, x, inputshape, device):
    #     x = self.blueBlockP1(x)
    #     x = self.blueSE(x, inputshape, device)
    #     x = self.postSE(x)
    #     return x
