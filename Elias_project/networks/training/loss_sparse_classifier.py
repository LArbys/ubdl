import os,sys

# torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

# sparse submanifold convnet library
import sparseconvnet as scn
# -------------------------------------------------------------------------
# HolePixelLoss
# This loss mimics nividia's pixelwise loss for holes (L1)
# used in the infill network
# how well does the network do in dead regions?
# -------------------------------------------------------------------------

# taken from torch.nn.modules.loss
def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"

class SparseClassifierLoss(nn.modules.loss._WeightedLoss):
    def __init__(self,weight=None, size_average=False, ignore_index=-100 ):
        super(SparseClassifierLoss,self).__init__(weight,size_average)
        self.ignore_index = ignore_index
        self.reduce = False
        self.size_average = size_average
        #self.mean = torch.mean.cuda()

    def forward(self,predict,true):
        """
        predict: (b,1,h,w) tensor with output from logsoftmax
        adc:  (b,h,w) tensor with true adc values
        """
        for i in range(0,len(true)):
            _assert_no_grad(true[i])
            true[i] = true[i].long()
        if predict == -1:
            return
        # print "size of predict: ",predict.size()
        # print "size of adc: ",adc.size()[0]
        # print("predict shape:",predict)
        # print("true_t shape:",true)
        # print("predict type:",type(predict))
        # print("true_t type:",type(true))
        # print("input_t type:",type(input))

        # 6 losses, each with weight tensors
        
        # print("size_average:",self.size_average)
        fl_weight_t = torch.tensor([1.0,1.0,1.0,1.0])
        # print("weights.dtyep",fl_weight_t.dtype)
        flavorsLoss=torch.nn.CrossEntropyLoss(weight=fl_weight_t, size_average=self.size_average)
        
        iT_weight_t = torch.tensor([1.0,1.0,1.0,1.0])
        interTypeLoss=torch.nn.CrossEntropyLoss(weight=iT_weight_t, size_average=self.size_average)
        
        nP_weight_t = torch.tensor([1.0,1.0,1.0,1.0])
        nProtonLoss=torch.nn.CrossEntropyLoss(weight=nP_weight_t, size_average=self.size_average)
        
        nCP_weight_t = torch.tensor([1.0,1.0,1.0,1.0])
        nCPionLoss=torch.nn.CrossEntropyLoss(weight=nCP_weight_t, size_average=self.size_average)
        
        nNP_weight_t = torch.tensor([1.0,1.0,1.0,1.0])
        nNPionLoss=torch.nn.CrossEntropyLoss(weight=nNP_weight_t, size_average=self.size_average)
        
        nN_weight_t = torch.tensor([1.0,1.0,1.0,1.0])
        nNeutronLoss=torch.nn.CrossEntropyLoss(weight=nN_weight_t, size_average=self.size_average)
        
        print("beginning loss calc:")
        
        # print("predict[0]:",predict[0])
        # print("true[0]:",true[0])
        # print("predict[0].dtype:",predict[0].dtype)
        # print("true[0].dtype:",true[0].dtype)
        # print("type(predict[0]):",type(predict[0]))
        # print("type(true[0]):",type(true[0]))
        fl_loss = flavorsLoss(predict[0],true[0])
        print("fl_loss:",fl_loss)
        
        iT_loss = interTypeLoss(predict[1],true[1])
        print("iT_loss:",iT_loss)
        
        nP_loss = nProtonLoss(predict[2],true[2])
        print("nP_loss:",nP_loss)
        
        nCP_loss = nCPionLoss(predict[3],true[3])
        print("nCP_loss:",nCP_loss)
        
        nNP_loss = nNPionLoss(predict[4],true[4])
        print("nNP_loss:",nNP_loss)
        
        nN_loss = nNeutronLoss(predict[5],true[5])
        print("nN_loss:",nN_loss)
        # nondeadweight = 1.0
        # deadnochargeweight = 500.0
        # deadlowchargeweight = 100.0
        # deadhighchargeweight = 100.0
        # deadhighestchargeweight = 100.0
        # 
        # goodch = (input > 0).float()
        # predictgood = goodch * predict
        # adcgood = goodch* adc
        # totnondead = goodch.sum().float()
        # if (totnondead == 0):
        #         totnondead = 1.0
        # nondeadloss = (CrossEntropyLoss(predictgood, adcgood)*nondeadweight)/totnondead
        # 
        # deadch = (input.eq(0)).float()
        # 
        # deadchhighestcharge = deadch * (adc > 70).float()
        # predictdeadhighestcharge = predict*deadchhighestcharge
        # adcdeadhighestcharge = adc*deadchhighestcharge
        # totdeadhighestcharge = deadchhighestcharge.sum().float()
        # if (totdeadhighestcharge == 0):
        #         totdeadhighestcharge = 1.0
        # deadhighestchargeloss = (CrossEntropyLoss(predictdeadhighestcharge,adcdeadhighestcharge)*deadhighestchargeweight)/totdeadhighestcharge
        # 
        # 
        # deadchhighcharge = deadch * (adc > 40).float()*(adc < 70).float()
        # predictdeadhighcharge = predict*deadchhighcharge
        # adcdeadhighcharge = adc*deadchhighcharge
        # totdeadhighcharge = deadchhighcharge.sum().float()
        # if (totdeadhighcharge == 0):
        #         totdeadhighcharge = 1.0
        # deadhighchargeloss = (CrossEntropyLoss(predictdeadhighcharge,adcdeadhighcharge)*deadhighchargeweight)/totdeadhighcharge
        # 
        # deadchlowcharge = deadch * (adc > 10).float() *(adc<40).float()
        # predictdeadlowcharge = predict*deadchlowcharge
        # adcdeadlowcharge = adc*deadchlowcharge
        # totdeadlowcharge = deadchlowcharge.sum().float()
        # if (totdeadlowcharge == 0):
        #         totdeadlowcharge = 1.0
        # deadlowchargeloss = (CrossEntropyLoss(predictdeadlowcharge,adcdeadlowcharge)*deadlowchargeweight)/totdeadlowcharge
        # 
        # deadchnocharge = deadch * (adc<10).float()
        # predictdeadnocharge = predict*deadchnocharge
        # adcdeadnocharge = adc*deadchnocharge
        # totdeadnocharge = deadchnocharge.sum().float()
        # if (totdeadnocharge == 0):
        #         totdeadnocharge = 1.0
        # deadnochargeloss = (CrossEntropyLoss(predictdeadnocharge,adcdeadnocharge)*deadnochargeweight)/totdeadnocharge

        totloss = fl_loss + iT_loss + nP_loss + nCP_loss + nNP_loss + nN_loss
        return fl_loss, iT_loss, nP_loss, nCP_loss, nNP_loss, nN_loss, totloss