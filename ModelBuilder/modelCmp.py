import torch
import torch.nn as nn
from torchsummary import summary
import torch.onnx
class model(nn.Module):
    def __init__(self, a, b, c): 
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.lst = [nn.Linear(2, 5), nn.Linear(2, 5), nn.Linear(10, 1), nn.ReLU()]
        self.layers = nn.ModuleList(self.lst).cuda()
    def forward(self, x):
        x1 = self.layers[0](x)
        x2 = self.layers[1](x)
        x3 = (torch.cat((x1,x2),-1))
        x4 = self.layers[2](x3)
        x5 = self.layers[3](x4)
        return x3
torchModel = model(1,2,3).cuda()
summary(torchModel,(32,2))