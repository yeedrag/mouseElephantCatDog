from blockClass import *
#from blockClass import Block
class Linear(Block):
    def __init__(self, layer, parents, index, args = {}):
        '''
            "args": {
            "inputSize": [batch_size, in_features],
            "outputSize": [batch_size, out_features],
            "bias": -> True or False
        }
        '''
        #cool
        super().__init__(layer, parents, index, args)
        assert len(self.inputSize) != 0
        assert len(self.outputSize) != 0
        self.net = nn.Linear(self.inputSize[-1], self.outputSize[-1], bias = self.bias) 
    def forward(self, x):
        #print(self.index, list(x.shape), self.inputSize)
        #self.updateShape()
        return self.net(x)
