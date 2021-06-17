import torch
from torch import nn
import sparseconvnet as scn

class SparseGlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(SparseGlobalAvgPool2d, self).__init__()
        # self.inputshape = inputshape
        # self.size = inputshape[0]*inputshape[1]
    
    def forward(self, x, inputshape):
        size = inputshape[0]*inputshape[1]
        # print("size:",size)
        # print("inputshape:",inputshape)
        shape = x.features.shape
        # print("x shape:",shape)

        x = self.sum(x,shape,inputshape)
        # print("SIZE:",size)
        x = self.div(x,size,shape)
        
        # print("x:",x)
        return x
    
    def sum(self,x, shape, inputshape):
        # print("x features type:",type(x.features))
        
        
        y = scn.SparseConvNetTensor(metadata=x.metadata,spatial_size=x.spatial_size)
        # print("x meta:",x.metadata)
        # print("y meta:",y.metadata)
        # y.metadata = x.metadata
        # y.spatial_size = self.inputshape
        z = []
        for i in range(0,shape[1]):
            sum = 0
            for j in range(0,shape[0]):
                sum += x.features[j,i]
            z.append(sum)
        y.features = torch.tensor(z)
        # print("y type:",type(y))
        # print("y type features:",type(y.features))
        # print("y shape:",y.features.shape)
        # print("y meta:",y.metadata)
        # print("y features:",y.features)
        # print("y:",y)
        return y
    
    def div(self,x,size,shape):
        # print("SHAPE:",shape)
        # print("SIZE:",size)
        for i in range(0,shape[1]):
            x.features[i] = x.features[i]/size
        
        return x
        
        
