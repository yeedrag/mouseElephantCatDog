from blockClass import *

class RNN(Block):
    def __init__(self, layer, parents, index, args = {}):
        '''
            "args": {
            "inputSize": [batch_size, in_features],
            "outputSize": [batch_size, out_features],
            "hiddenSize": int,
            "numLayers": int,
            "nonLinearity ": "tanh" or "relu"
            "dropout": int(0~1)
            "bidirectional" bool

        }
        '''
        super().__init__(layer, parents, index, args)
        
    def forward(self, x):
        
        return self.net(x)