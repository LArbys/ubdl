from torch import nn
import sparseconvnet as scn
import SparseGlobalAvgPool2d as gavg


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = gavg.SparseGlobalAvgPool2d()
        self.fc = scn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x, inputshape):
        b, c  = x.features.size()
        # print("b:",b)
        # print("c:",c)
        y = self.avg_pool(x, inputshape)
        # print("y:",y)
        # print("TYPE of features:",y.features.dtype)
        y.features = self.fc(y.features)
        # print("x type before returning se layer forward pass:",type(x))
        x.features = x.features * y.features.expand_as(x.features)
        return x
