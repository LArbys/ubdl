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

    def forward(self,predict,true,device):        
        # 6 losses, each with weight tensors
        fl_weight_t = torch.tensor([1.0,1.0,1.0],device=device)
        flavorsLoss=torch.nn.CrossEntropyLoss(weight=fl_weight_t, size_average=self.size_average)
        
        iT_weight_t = torch.tensor([1.0,1.0,1.0,1.0],device=device)
        interTypeLoss=torch.nn.CrossEntropyLoss(weight=iT_weight_t, size_average=self.size_average)
        
        nP_weight_t = torch.tensor([1.0,1.0,1.0,1.0],device=device)
        nProtonLoss=torch.nn.CrossEntropyLoss(weight=nP_weight_t, size_average=self.size_average)
        
        nCP_weight_t = torch.tensor([1.0,1.0,1.0,1.0],device=device)
        nCPionLoss=torch.nn.CrossEntropyLoss(weight=nCP_weight_t, size_average=self.size_average)
        
        nNP_weight_t = torch.tensor([1.0,1.0,1.0,1.0],device=device)
        nNPionLoss=torch.nn.CrossEntropyLoss(weight=nNP_weight_t, size_average=self.size_average)
        
        nN_weight_t = torch.tensor([1.0,1.0,1.0,1.0],device=device)
        nNeutronLoss=torch.nn.CrossEntropyLoss(weight=nN_weight_t, size_average=self.size_average)
        
        print("beginning loss calc:")
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

        totloss = fl_loss + iT_loss + nP_loss + nCP_loss + nNP_loss + nN_loss
        print("total loss:",totloss)
        
        return fl_loss, iT_loss, nP_loss, nCP_loss, nNP_loss, nN_loss, totloss