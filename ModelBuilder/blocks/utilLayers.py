from blockClass import *

class Input(Block):
    def __init__(self, layer, parents, index, args = {}):
        super().__init__(layer, parents, index, args)
    def forward(self, x):
        #self.updateShape()
        return x
class Concat(Block):
    def __init__(self, layer, parents, index, args = {}):
        # Required args: dim
        '''
            "args": {
            "inputSize": [],
            "outputSize": [],
            "dim": int
        }
        '''
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

class flatten(Block):
    def __init__(self, layer, parents, index, args = {}):
        '''
            "args": {
            "inputSize": [],
            "outputSize": [],
            "start_dim": int(initial 1),
            "end_dim": int(initial -1)
        }
        '''
        super().__init__(layer, parents, index, args)
        self.inputSize  = layer[parents[0]].outputSize
        for i in range(0,self.start_dim):
            self.outputSize.append(self.inputSize[i])
        self.outputSize.append(1)
        for i in range(self.start_dim,self.end_dim+1):
            self.outputSize[-1] *= self.inputSize[i]
        for i in range(self.end_dim+1,len(self.inputSize)):
            self.outputSize.append(self.inputSize[i],)
    def forward(self, x):
        out = torch.flatten(x,start_dim = self.start_dim, end_dim = self.end_dim)
        return out