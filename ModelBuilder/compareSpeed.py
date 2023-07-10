import torch
import torch.nn as nn
from blockDictionary import callBlock
from blocks.blockClass import Block
import time
class Dummy():
    def __init__(self, outputSize):
        self.outputSize = outputSize
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
    
    
layers = [Dummy([32, 2]), Dummy([32, 2]), Dummy([32, 2])]
parents = [0, 1, 2]
args = {"inputSize": [],"outputSize": [],"dim": 1}

start = time.time()
print("--- Concat 10000 times ---")
for i in range(10000):
    concatCalc = callBlock["Concat"](layers, parents, 3, args)
print("--- Calculate took %s seconds ---" % (time.time() - start))
start = time.time()
for i in range(10000):
    concat = concatDummy(layers, parents, 3, args)
print("--- Dummy took %s seconds ---" % (time.time() - start))
