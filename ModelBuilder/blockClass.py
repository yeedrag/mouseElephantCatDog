import torch.nn as nn
import torch
# maybe can change inputSize into index, and just search with that!
class Block(nn.Module):
    def __init__(self, layer, parents, index, inputSize = [], outputSize = []):
        super().__init__()
        self.layer = layer
        self.parents = parents # array
        self.index = index
        if(len(inputSize) == 0 and len(self.parents) != 0):
            self.inputSize = self.layer[parents[0]].outputSize
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

# Super important! change from searching in data to searching in xArray instead!!!!
# Or else for concat, relu, conv.... with no set output will assert error for self.inputsize != 0
# fix tomorrow!!!!!!! its 3 a.m and I should sleep!!!!

class Linear(Block):
    def __init__(self, layer, parents, index, inputSize = [], outputSize = [], bias = True):
        super().__init__(layer, parents, index, inputSize, outputSize)
        self.bias = bias
        assert len(self.inputSize) != 0
        assert len(self.outputSize) != 0
        self.net = nn.Linear(self.inputSize[-1], self.outputSize[-1], bias = self.bias)
    def forward(self, x):
        #self.updateShape()
        return self.net(x)

class Activation(Block):
    def __init__(self, layer, parents, index, inputSize = [], outputSize = [], mode = "ReLU"): # type is reserved lmao
        super().__init__(layer, parents, index, inputSize, outputSize)
        match mode:
            case "ReLU":
                self.net = nn.ReLU()
            case "LeakyReLU":
                self.net = nn.LeakyReLU()
            case "Sigmoid":
                self.net = nn.Sigmoid()
            case "Tanh":
                self.net = nn.Tanh()
            case _: # other
                raise NotImplementedError
        #self.updateShape()
    def forward(self, x):
        return self.net(x)
    
class Input(Block):
    def __init__(self, layer, parents, index, inputSize = [], outputSize = []):
        super().__init__(layer, parents, index, inputSize, outputSize)
    def forward(self, x):
        #self.updateShape()
        return x
# We should calculate inputsize and outputsize in __init__ .....
class Concat(Block):
    def __init__(self, layer, parents, index, inputSize = [], outputSize = [], dim = 1):
        super().__init__(layer, parents, index, inputSize, outputSize)
        self.ref = self.layer[parents[0]].outputSize # choose 1st parent as reference
        self.dim = dim if dim >= 0 else len(self.ref) - abs(dim) # convert to index >= 0
        assert len(self.ref) > dim and dim > 0 # check if dim is valid
        self.inputSize = [self.ref]
        for idx in range(1, len(parents)):
            size = self.layer[idx].outputSize
            self.inputSize.append(size)
            for i in range(0, len(self.ref)): # check if values beside dim is correct
                if i == dim:
                    self.ref[i] += size[i]
                else:
                    assert size[i] == self.ref[i]
        self.outputSize = self.ref
        # or maybe consider just torch.rand(size) and actually try concat?
    def forward(self, x): # x is a list of multiple tensors
        out = torch.cat(x, dim = self.dim)
        #update json 
        #self.inputSize = [item.shape for item in x]
        #self.outputSize = out.shape
        #self.updateShape()
        return out

