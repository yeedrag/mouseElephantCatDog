import torch.nn as nn
import torch
# maybe can change inputSize into index, and just search with that!
class Block(nn.Module):
    def __init__(self, data, index, inputSize = [], outputSize = []):
        super().__init__()
        self.data = data  
        self.index = index
        if(len(inputSize) == 0 and len(data[index]["parent"]) != 0):
            self.inputSize = self.data[self.data[self.index]["parent"][0]]["args"]["outputSize"]
        else:
            self.inputSize = inputSize
        self.outputSize = outputSize # most should have outputSizes....except maybe convs
    def initWeight(self):
        raise NotImplementedError
    def updateShape(self):
        # bug :(
        self.data[self.index]["inputSize"] = self.inputSize
        self.data[self.index]["outputSize"] = self.outputSize
    def forward(self, x):
        return x
# maybe can name each layer with layer name + index?
class Linear(Block):
    def __init__(self, data, index, inputSize = [], outputSize = [], bias = True):
        super().__init__(data, index, inputSize, outputSize)
        self.bias = bias
        assert len(self.inputSize) != 0
        assert len(self.outputSize) != 0
        self.net = nn.Linear(self.inputSize[-1], self.outputSize[-1], bias = self.bias)
    def forward(self, x):
        #self.updateShape()
        return self.net(x)

class ReLU(Block):
    def __init__(self, data, index, inputSize = [], outputSize = []):
        super().__init__(data, index, inputSize, outputSize)
        self.net = nn.ReLU()
        #self.updateShape()
    def forward(self, x):
        return self.net(x)
    
class Input(Block):
    def __init__(self, data, index, inputSize = [], outputSize = []):
        super().__init__(data, index, inputSize, outputSize)
    def forward(self, x):
        #self.updateShape()
        return x
# We should calculate inputsize and outputsize in __init__ .....
class Concat(Block):
    def __init__(self, data, index, inputSize = [], outputSize = [], dim = 1):
        super().__init__(data, index, inputSize, outputSize)
        self.dim = dim
    def forward(self, x): # x is a list of multiple tensors
        out = torch.cat(x, dim = self.dim)
        #update json 
        self.inputSize = [item.shape for item in x]
        self.outputSize = out.shape
        #self.updateShape()
        return out

