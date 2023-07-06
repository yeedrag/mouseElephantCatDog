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
        self.layers = nn.ModuleList([nn.Linear(2, 5), nn.Linear(2, 5), nn.Linear(10, 1)]).cuda()
    def forward(self, x):
        x1 = self.layers[0](x)
        x2 = self.layers[1](x)
        x3 = self.layers[2](torch.cat((x1,x2),-1))
        return x3
torchModel = model(1,2,3).cuda()
summary(torchModel,(32,2))
dummyInput = torch.rand([32,2]).cuda()
torch.onnx.export(torchModel,dummyInput,"modelCmp.onnx")