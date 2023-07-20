from blockClass import *

class Activation(Block):
    def __init__(self, layer, parents, index, args = {}): # type is reserved lmao
        '''
            "args": {
            "inputSize": [batch_size, in_features],
            "outputSize": [batch_size, out_features],
            "mode": layerName
        }
        '''
        super().__init__(layer, parents, index, args)
        self.outputSize = self.inputSize
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