from blockClass import *

class Dropout(Block):
    def __init__(self, layer, parents, index, args = {}):
        '''
            "args": {
            "inputSize": [batch_size, in_features],
            "outputSize": [batch_size, out_features],
            "p": int,
            "inplace": bool
        }
        '''
        super().__init__(layer, parents, index, args)
        self.outputSize = self.inputSize
        self.net = nn.Dropout(self.p, self.inplace)
    def forward(self,x):
        return self.net(x)

#dummy = torch.randn([5,5])
#print(dummy)
#drop = Dropout(1, 1, 1,{"inputSize": [5,5], "outputSize": [5,5], "p":0.5, "implace": False})
#print(drop(dummy))