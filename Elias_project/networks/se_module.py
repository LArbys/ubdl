from torch import nn
import sparseconvnet as scn
import SparseGlobalAvgPool2d as gavg


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        
           
    # DENSE IMPLEMENTATION, CHECK AGAINST ORIGINAL TO PUT IT BACK
    # def __init__(self, channel, reduction=16):
    #     super(SELayer, self).__init__()
    #     self.avg_pool = nn.AdaptiveAvgPool2d(1)
    #     self.fc = nn.Sequential(
    #         nn.Linear(channel, channel // reduction, bias=False),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(channel // reduction, channel, bias=False),
    #         nn.Sigmoid()
    #     )
    # 
    # def forward(self, x):
    #     b, c, _, _  = x.size()
    #     y = self.avg_pool(x).view(b, c)
    #     y = self.fc(y).view(b, c, 1, 1)
    #     return x * y.expand_as(x)
    
    
    # SPARSE IMPLEMENTATION
    # def __init__(self, channel, reduction=16):
    #     super(SELayer, self).__init__()
    #     self.avg_pool = gavg.SparseGlobalAvgPool2d()
    #     self.fc = scn.Sequential(
    #         nn.Linear(channel, channel // reduction, bias=False),
    #         nn.ReLU(),
    #         nn.Linear(channel // reduction, channel, bias=False),
    #         nn.Sigmoid()
    #     )
    # 
    # def forward(self, x, inputshape):
    #     b, c  = x.features.size()
    #     # print("b:",b)
    #     # print("c:",c)
    #     y = self.avg_pool(x, inputshape)
    #     # print("y:",y)
    #     # print("TYPE of features:",y.features.dtype)
    #     y.features = self.fc(y.features)
    #     # print("x type before returning se layer forward pass:",type(x))
    #     x.features = x.features * y.features.expand_as(x.features)
    #     return x

    
    
    
    
    
    