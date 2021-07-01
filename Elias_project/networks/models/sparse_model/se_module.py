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
        y = self.avg_pool(x, inputshape)
        y.features = self.fc(y.features)
        x.features = x.features * y.features.expand_as(x.features)
        return x
