import torch
from torch import nn
import sparseconvnet as scn

class SparseGlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(SparseGlobalAvgPool2d, self).__init__()
    
    def forward(self, x, inputshape, device):
        size = inputshape[0]*inputshape[1]
        shape = x.features.shape
        x = self.sum(x,shape,inputshape,device)
        x = self.div(x,size,shape)
        return x
    
    def sum(self,x, shape, inputshape,device):
        y = scn.SparseConvNetTensor(metadata=x.metadata,spatial_size=x.spatial_size)
        z = []
        for i in range(0,shape[1]):
            sum = 0
            for j in range(0,shape[0]):
                sum += x.features[j,i]
            z.append(sum)
        y.features = torch.tensor(z,device=device)
        return y
    
    def div(self,x,size,shape):
        for i in range(0,shape[1]):
            x.features[i] = x.features[i]/size
        return x        
        
