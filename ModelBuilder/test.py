import torch.nn as nn
import torch
from torchsummary import summary
class conv(nn.Module):
    def __init__(self, inp, outp, ker):
        super().__init__()
        self.inp = inp
        self.outp = outp
        self.ker = ker
        self.net = nn.Conv2d(self.inp, self.outp, self.ker)
    def forward(self, x):
        return self.net(x)
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.con = conv(10, 100, 3)
        self.net = nn.Sequential(self.con, nn.ReLU()).cuda()
    def forward(self, x):
        return self.net(x)
#torchModel = model()
#print(torchModel)
#summary(torchModel,dummyInputCNN)
#torch.onnx.export(torchModel,dummyInputCNN,"modelCNNtest.onnx")
