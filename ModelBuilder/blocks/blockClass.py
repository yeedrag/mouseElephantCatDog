import torch.nn as nn
import torch
import copy
# maybe can change inputSize into index, and just search with that!
class Block(nn.Module):
    def __init__(self, layer, parents, index, args = {}):
        super().__init__()
        self.mapDict(args)
        self.args = args
        self.layer = layer
        self.parents = parents # array
        self.index = index
        if(len(self.inputSize) == 0 and len(self.parents) != 0):
            self.inputSize = self.layer[parents[0]].outputSize
        if(len(self.outputSize) == 0):
            self.outputSize = self.inputSize # still should clarify in indiv blocks
        # most should have outputSizes....except maybe convs
    def mapDict(self, args): 
        for k, v in args.items():
            setattr(self, k, v) # self.k = v
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
'''
class Linear(Block):
    def __init__(self, layer, parents, index, args = {}):
        super().__init__(layer, parents, index, args)
        assert len(self.inputSize) != 0
        assert len(self.outputSize) != 0
        self.net = nn.Linear(self.inputSize[-1], self.outputSize[-1], bias = self.bias)
    def forward(self, x):
        #self.updateShape()
        return self.net(x)


class Activation(Block):
    def __init__(self, layer, parents, index, args = {}): # type is reserved lmao
        super().__init__(layer, parents, index, args)
        match self.mode:
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
    def __init__(self, layer, parents, index, args = {}):
        super().__init__(layer, parents, index, args)
    def forward(self, x):
        #self.updateShape()
        return x

# We should calculate inputsize and outputsize in __init__ .....
class Concat(Block):
    def __init__(self, layer, parents, index, args = {}):
        # Required args: dim
        super().__init__(layer, parents, index, args)
        self.ref = copy.copy(self.layer[parents[0]].outputSize) # choose 1st parent as reference, I hate python
        self.dim = self.dim if self.dim >= 0 else len(self.ref) - abs(self.dim) # convert to index >= 0 
        assert len(self.ref) > self.dim and self.dim > 0 # check if dim is valid
        self.inputSize = [self.ref]
        for idx in range(1, len(parents)):
            size = self.layer[idx].outputSize
            self.inputSize.append(size)
            for i in range(0, len(self.ref)): # check if values beside dim is correct
                if i == self.dim:
                    self.ref[i] += size[i]
                else:
                    assert size[i] == self.ref[i]
        self.outputSize = copy.copy(self.ref)
        # or maybe consider just torch.rand(size) and actually try concat?
    def forward(self, x): # x is a list of multiple tensors
        out = torch.cat(x, dim = self.dim)
        #update json 
        #self.inputSize = [item.shape for item in x]
        #self.outputSize = out.shape
        #self.updateShape()
        return out
class concatDummy(Block):
    def __init__(self, layer, parents, index, args = {}):
        # Required args: dim
        super().__init__(layer, parents, index, args)
        self.inputSize  = [layer[parent].outputSize for parent in parents]
        Dummies = [torch.rand(sz) for sz in self.inputSize]
        dummyOutput = torch.cat(Dummies, dim = self.dim)
        self.outputSize = list(dummyOutput.shape)
        # or maybe consider just torch.rand(size) and actually try concat?
    def forward(self, x): # x is a list of multiple tensors
        out = torch.cat(x, dim = self.dim)
        return out
'''
class Conv2d(Block):
    def __init__(self, layer, parents, index, args = {}):
        super().__init__(layer, parents, index, args = {})

    def forward(self, x):
        
        return self.net(x)
