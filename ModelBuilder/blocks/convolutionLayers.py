from blockClass import *
import time
class Conv(Block): # nn.conv2d
    def __init__(self, layer, parents, index, args = {}):
        '''
            "args": {
            "inputSize": [N, C_in, H, W],
            "outputSize": [N, C_out, H_out, W_out], # will be calculated
            "outputChannels": int,
            "kernelSize": [height, width],
            "stride": [height, width],
            "padding": [height, width] or str: "same", "valid",
            "dilation": [height, width],
            "groups": int,
            "bias": bool,
            "paddingMode": str, 'zeros', 'reflect', 'replicate' or 'circular'
        }
        '''
        super().__init__(layer, parents, index, args)  
        if(isinstance(self.padding, str)):
            if(self.padding == "same"):
                self.outputSize = copy.copy(self.inputSize)
                self.outputSize[1] = self.outputChannels

                # work around for onnx export problem, https://github.com/pytorch/pytorch/issues/68880
                # solve is from tensorflow. https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/4
                if(self.inputSize[2] % self.stride[0] == 0):
                    padAlongHeight = max(self.kernelSize[0] - self.stride[0], 0)
                else:
                    padAlongHeight = max(self.kernelSize[0] - (self.inputSize[2] % self.stride[0]), 0)
                if(self.inputSize[3] % self.stride[1] == 0):
                    padAlongWidth = max(self.kernelSize[1] - self.stride[1], 0)
                else:
                    padAlongWidth = max(self.kernelSize[1] - (self.inputSize[3] % self.stride[1]), 0)
                self.padding = [padAlongHeight, padAlongWidth]
            elif(self.padding == "valid"): 
                self.padding = [0, 0] # no padding
        if(isinstance(self.padding, list)):
            self.outputHeight = (self.inputSize[2] + (2 * self.padding[0]) - self.dilation[0] * (self.kernelSize[0] - 1) - 1 + self.stride[0]) // self.stride[0]
            self.outputWidth = (self.inputSize[3] + (2 * self.padding[1]) - self.dilation[1] * (self.kernelSize[1] - 1) - 1 + self.stride[1]) // self.stride[1]
            self.outputSize = [self.inputSize[0], self.outputChannels, self.outputHeight, self.outputWidth]
        #self.net = nn.Conv2d(self.inputSize[1], self.outputChannels, self.kernelSize, self.stride, self.padding,
        #                           self.dilation, self.groups, self.bias, self.paddingMode)
        if(isinstance(self.padding, str) and self.padding == "same"):
            assert self.outputSize == self.inputSize
        # I set to 0 now will error if pad = "same"
        self.net = nn.Conv2d(self.inputSize[1], self.outputChannels, self.kernelSize, self.stride, self.padding,
                                    self.dilation, self.groups, self.bias)
        # 這到底沙小扣
        #別人也是這樣 哈哈
    def forward(self, x):
        #print(self.index, self.inputSize, list(x.shape))
        out = self.net(x)
        #print(self.index, self.outputSize, list(out.shape))
        #assert(self.outputSize == list(out.shape))
        return out
class ConvDummy(Block): # nn.conv2d, slow!
    def __init__(self, layer, parents, index, args = {}):
        super().__init__(layer, parents, index, args)
        #self.net = nn.Conv2d(self.inputSize[1], self.outputChannels, self.kernelSize, self.stride, self.padding,
        #                     self.dilation, self.groups, self.bias, self.paddingMode)
        # Interestingly, onnx does not support padding lol
        self.net = nn.Conv2d(self.inputSize[1], self.outputChannels, self.kernelSize)
        dummy = torch.rand(self.inputSize)
        dummyOutput = self.net(dummy)
        self.outputSize = list(dummyOutput.shape)
        # 這到底沙小扣
        #別人也是這樣 哈哈
    def forward(self, x):
        return self.net(x)

class Pooling(Block):
    def __init__(self, layer, parents, index, args = {}):
        '''
            "args": {
            "inputSize": [N, C_in, H, W],
            "outputSize": [N, C_out, H_out, W_out], # will be calculated
            "mode": str, "max", "avg", "adaptiveMax", "adaptiveAvg"
            "outputSizeAdapt": [H_out, W_out], [0, 0] if mode isn't "adaptiveMax" or "adaptiveAvg"
            "kernelSize": [height, width],
            "stride": [height, width],
            "padding": [height, width], no string!
            "dilation": [height, width], [1, 1] if mode isn't max
        }
        '''
        super().__init__(layer, parents, index, args)  
        # onnx does not support adaptive.... find other way to global pooling....
        if self.mode == "adaptiveMax":
            self.net = nn.AdaptiveMaxPool2d(self.outputSizeAdapt)
            self.outputSize = copy.copy(self.inputSize)
            self.outputSize[-2:] = self.outputSizeAdapt
        elif self.mode == "adaptiveAvg":
            self.net = nn.AdaptiveAvgPool2d(self.outputSizeAdapt)
            self.outputSize = copy.copy(self.inputSize)
            self.outputSize[-2:] = self.outputSizeAdapt
        else:
            if(isinstance(self.padding, list)):
                self.outputHeight = (self.inputSize[2] + (2 * self.padding[0]) - self.dilation[0] * (self.kernelSize[0] - 1) - 1 + self.stride[0]) // self.stride[0]
                self.outputWidth = (self.inputSize[3] + (2 * self.padding[1]) - self.dilation[1] * (self.kernelSize[1] - 1) - 1 + self.stride[1]) // self.stride[1]
                self.outputSize = [self.inputSize[0], self.inputSize[1], self.outputHeight, self.outputWidth]
            if self.mode == "max":
                self.net = nn.MaxPool2d(self.kernelSize, self.stride, self.padding, self.dilation)
            elif self.mode == "avg":
                # avg pool doens't have dilation for some reason
                self.net = nn.AvgPool2d(self.kernelSize, self.stride, self.padding) 
    def forward(self, x):
        #print(self.index, list(x.shape), self.inputSize)
        out = self.net(x)
        #print(self.index, self.outputSize, list(out.shape))
        return self.net(x)
if __name__ == "__main__":
    # test speed, or maybe put in another place zzz
    start = time.time()
    times = 100
    print("--- Conv {} times ---".format(times))
    for i in range(times):
        con = Conv(1, 1, 1, {
                    "inputSize": [32, 10, 32, 32],
                    "outputSize": [],
                    "outputChannels": 100,
                    "kernelSize": [3, 3],
                    "stride": [1, 1],
                    "padding": "same",
                    "dilation": [1, 1],
                    "groups": 1,
                    "bias": True,
                    "paddingMode": "zeros"
                }
                )
    print("--- Calculate took %s seconds ---" % (time.time() - start))
    start = time.time()
    for i in range(times):
        conDummy = ConvDummy(1, 1, 1, {
                    "inputSize": [32, 10, 32, 32],
                    "outputSize": [],
                    "outputChannels": 100,
                    "kernelSize": [3, 3],
                    "stride": [1, 1],
                    "padding": "same",
                    "dilation": [1, 1],
                    "groups": 1,
                    "bias": True,
                    "paddingMode": "zeros"
                })
    print("--- Dummy took %s seconds ---" % (time.time() - start))