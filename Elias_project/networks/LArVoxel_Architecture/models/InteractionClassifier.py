from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import sparseconvnet as scn

import MinkowskiEngine as ME
# from .resnetinstance_block import BasicBlockInstanceNorm
from minkencodedecode import MinkEncode34C,MinkDecode34C

class InteractionClassifier(nn.Module):

    def __init__(self,input_feats=1,
                 embed_dim=512,
                 out_num_classes=7):
        """
        Network consists of 
          1) a ResNet encoder with a stem+6 layers. each layer downsamples.
          2) a 3 layer convolution module that mixes features from each downsampled resnet layer
          3) a global max and ave pooling layer that each merges the features at all pixels into one feature vector
          3) an MLP that takes the concat of the pooling layers into the class predictions
        parameters
        -----------
        """
        super(InteractionClassifier,self).__init__()

        # Number of dimensions of our input tensor
        self.D = 2

        # Resnet encoder
        self.encoder = MinkEncode34C(in_channels=input_feats, out_channels=1, D=self.D)

        # sum the channels at each output layer of the encoder
        totchannels = self.encoder.INIT_DIM # the stem output channels
        for nch in self.encoder.PLANES:
            totchannels += nch # output channels of each layer
        #print("tot channels: ",totchannels)

        # the convoutional layers that mixes features from all the resnet layers
        multiscale_layers = []
        multiscale_layers.append(
            nn.Sequential(
                ME.MinkowskiConvolution(
                    totchannels,
                    embed_dim//4,
                    kernel_size=3,
                    stride=2,
                    dimension=self.D,
                ),
                ME.MinkowskiInstanceNorm(embed_dim//4),
                ME.MinkowskiLeakyReLU()))
        multiscale_layers.append(
            nn.Sequential(
                ME.MinkowskiConvolution(
                    embed_dim//4,
                    embed_dim//2,
                    kernel_size=3,
                    stride=2,
                    dimension=self.D,
                ),
                ME.MinkowskiInstanceNorm(embed_dim//2),
                ME.MinkowskiLeakyReLU() ))
        multiscale_layers.append(
            nn.Sequential(
                ME.MinkowskiConvolution(
                    embed_dim//2,
                    embed_dim,
                    kernel_size=3,
                    stride=2,
                    dimension=self.D,
                ),
                ME.MinkowskiInstanceNorm(embed_dim),
                ME.MinkowskiLeakyReLU(),
            ) )
        self.multiscale_conv = nn.Sequential(*multiscale_layers)

        # the global pooling layers
        # location within the dense tensor is lost
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

        # the final MLP
        self.final = nn.Sequential(
            self.get_mlp_block(embed_dim * 2 * 3, 512),
            ME.MinkowskiDropout(),
            self.get_mlp_block(512, 512),
            # ME.MinkowskiLinear(512, out_num_classes, bias=True),
            #ME.MinkowskiLinear(embed_dim*2, out_num_classes, bias=True),
        )

        self.flavors = ME.MinkowskiLinear(512, 3, bias=True)
        self.interactionType = ME.MinkowskiLinear(512, 4, bias=True)
        self.nProton = ME.MinkowskiLinear(512, 4, bias=True)
        self.nCPion = ME.MinkowskiLinear(512, 4, bias=True)
        self.nNPion = ME.MinkowskiLinear(512, 4, bias=True)
        self.nNeutron = ME.MinkowskiLinear(512, 4, bias=True)

        self.SoftMax = nn.Softmax(1)

        self.weight_initialization()

    def get_mlp_block(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiInstanceNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
                
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
            if isinstance(m, ME.MinkowskiStableInstanceNorm):
                m.reset_parameters()

    
    def forward(self, input): #  : ME.TensorField 
        # print("input:",input)
        toConcat = []
        for i in range(3):
            inSparse = input[i].sparse()
            # print("inSparse:",inSparse)
            encoder_out   = self.encoder(inSparse)
            # print("encoder_out:",encoder_out)
            y = [ x.slice(input[i]) for x in encoder_out ]
            y = ME.cat(y).sparse()
            y = self.multiscale_conv(y)
            y = ME.cat( self.global_max_pool(y),self.global_avg_pool(y) )
            # print("y.shape:",y.shape)
            toConcat.append(y)
            
            # print("y.coordinate_map_key:",y.coordinate_map_key)

            # print("y.coordinate_manager:",y.coordinate_manager)

        # print("xinput:",xinput)
        # xsparse = xinput.sparse()
        # print("xsparse:",xsparse)
        # encoder_out   = self.encoder(xsparse)

        # we slice, which broadcasts pooled features back to original resolution (I think)
        # y = [ x.slice(xinput) for x in encoder_out ]
        # print("*toConcat:", *toConcat)
        # y = ME.cat(*toConcat).sparse()
        # print("y.shape:",y.shape)
        # y = self.multiscale_conv(y)
        # print("y.shape:",y.shape)
        # y = ME.cat( self.global_max_pool(y),self.global_avg_pool(y) )
        y = ME.cat(*toConcat)
        # print("y.shape:",y.shape)
        # print("y:",y)
        #print("after pool: ",y.F)
        y = self.final(y)
        #print("after final conv: ",y.F)
        fl = self.flavors(y)

        # inter = self.interactionType(y)
        # nP = self.nProton(y)
        # nCP = self.nCPion(y)
        # nNP = self.nNPion(y)
        # nN = self.nNeutron(y)

        # out = [fl.F, inter.F, nP.F, nCP.F, nNP.F, nN.F]

        # return out
        return [fl.F]

        # TODO: Simplify to just flavors
        # TODO: check gradients (nan, crazy values)
        # TODO: send in one entry and make sure the gradients are updating
        # TODO: check that a perfect answer produces a loss of 0

    def deploy(self, input): #  : ME.TensorField 
        # print("input:",input)
        toConcat = []
        for i in range(3):
            inSparse = input[i].sparse()
            # print("inSparse:",inSparse)
            encoder_out   = self.encoder(inSparse)
            # print("encoder_out:",encoder_out)
            y = [ x.slice(input[i]) for x in encoder_out ]
            y = ME.cat(y).sparse()
            y = self.multiscale_conv(y)
            y = ME.cat( self.global_max_pool(y),self.global_avg_pool(y) )
            # print("y.shape:",y.shape)
            toConcat.append(y)
            
            # print("y.coordinate_map_key:",y.coordinate_map_key)

            # print("y.coordinate_manager:",y.coordinate_manager)

        # print("xinput:",xinput)
        # xsparse = xinput.sparse()
        # print("xsparse:",xsparse)
        # encoder_out   = self.encoder(xsparse)

        # we slice, which broadcasts pooled features back to original resolution (I think)
        # y = [ x.slice(xinput) for x in encoder_out ]
        # print("*toConcat:", *toConcat)
        # y = ME.cat(*toConcat).sparse()
        # print("y.shape:",y.shape)
        # y = self.multiscale_conv(y)
        # print("y.shape:",y.shape)
        # y = ME.cat( self.global_max_pool(y),self.global_avg_pool(y) )
        y = ME.cat(*toConcat)
        # print("y.shape:",y.shape)
        # print("y:",y)
        #print("after pool: ",y.F)
        y = self.final(y)

        fl = self.flavors(y)
        fl = self.SoftMax(fl.F)

        # inter = self.interactionType(y)
        # inter = self.SoftMax(inter.F)
        
        # nP = self.nProton(y)
        # nP = self.SoftMax(nP.F)

        # nCP = self.nCPion(y)
        # nCP = self.SoftMax(nCP.F)

        # nNP = self.nNPion(y)
        # nNP = self.SoftMax(nNP.F)

        # nN = self.nNeutron(y)
        # nN = self.SoftMax(nN.F)

        # out = [fl, inter, nP, nCP, nNP, nN]

        # return out

        return [fl]