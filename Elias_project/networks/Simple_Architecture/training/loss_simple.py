import os,sys

# torch
import torch
import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim
# import torch.utils.data
# import torch.utils.data.distributed
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models
import torch.nn.functional as F

# taken from torch.nn.modules.loss
def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"

class SparseClassifierLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, device, weight=None, size_average=False, ignore_index=-100 ):
        super(SparseClassifierLoss,self).__init__(weight,size_average)
        # self.ignore_index = ignore_index
        # self.reduce = False
        self.size_average = size_average
        self.device = device
        #self.mean = torch.mean.cuda()

    def forward(self,predict,truth):
        # print("predict:",predict)
        # print("truth", truth)
        # 6 losses, each with weight tensors
        fl_weight_t = torch.tensor([1.0,1.0,1.0], device=self.device)
        flavorsLoss = torch.nn.CrossEntropyLoss(weight=fl_weight_t, size_average=self.size_average)
        
        # print("predict:", predict)
        # print("truth[0]:", truth[0])
        fl_loss = flavorsLoss(predict, truth[0])
        # print("fl_loss:",fl_loss)

        # iT_weight_t = torch.tensor([1.0,1.0,1.0,1.0],device=self.device)
        # interTypeLoss = torch.nn.CrossEntropyLoss(weight=iT_weight_t, size_average=self.size_average)
        
        # nP_weight_t = torch.tensor([1.0,1.0,1.0,1.0],device=self.device)
        # nProtonLoss = torch.nn.CrossEntropyLoss(weight=nP_weight_t, size_average=self.size_average)
        
        # nCP_weight_t = torch.tensor([1.0,1.0,1.0,1.0],device=self.device)
        # nCPionLoss = torch.nn.CrossEntropyLoss(weight=nCP_weight_t, size_average=self.size_average)
        
        # nNP_weight_t = torch.tensor([1.0,1.0,1.0,1.0],device=self.device)
        # nNPionLoss = torch.nn.CrossEntropyLoss(weight=nNP_weight_t, size_average=self.size_average)
        
        # nN_weight_t = torch.tensor([1.0,1.0,1.0,1.0],device=self.device)
        # nNeutronLoss = torch.nn.CrossEntropyLoss(weight=nN_weight_t, size_average=self.size_average)
        
        # print("Beginning Loss")
        # fl_loss = flavorsLoss(predict[0],truth[0])
        # print("fl_loss:",fl_loss)
        # # print("fl_loss:",fl_loss.device)
        
        # iT_loss = interTypeLoss(predict[1],truth[1])
        # print("iT_loss:",iT_loss)
        # # print("iT_loss:",iT_loss.device)
        
        # nP_loss = nProtonLoss(predict[2],truth[2])
        # print("nP_loss:",nP_loss)
        # # print("nP_loss:",nP_loss.device)
        
        # nCP_loss = nCPionLoss(predict[3],truth[3])
        # print("nCP_loss:",nCP_loss)
        # # print("nCP_loss:",nCP_loss.device)
        
        # nNP_loss = nNPionLoss(predict[4],truth[4])
        # print("nNP_loss:",nNP_loss)
        # # print("nNP_loss:",nNP_loss.device)
        
        # nN_loss = nNeutronLoss(predict[5],truth[5])
        # print("nN_loss:",nN_loss)
        # # print("nN_loss:",nN_loss.device)
        
        # totloss = fl_loss + iT_loss + nP_loss + nCP_loss + nNP_loss + nN_loss
        # print("total loss:",totloss)
        # # print("total loss:",totloss.device)
        
        # return fl_loss, iT_loss, nP_loss, nCP_loss, nNP_loss, nN_loss, totloss
        return fl_loss