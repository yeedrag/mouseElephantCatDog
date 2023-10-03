import torch.nn as nn
import torch
from torchsummary import summary

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(16,16,False).cuda()
        self.layer1 = nn.Linear(16,1,True).cuda()
    def forward(self, x):
        x = self.input(x)
        x = self.layer1(x)
        return x
torchmodel = model()
summary(torchmodel,input_size=(16,),device="cuda")

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
