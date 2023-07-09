import torch
import torch.nn as nn
from blockClass import Concat
import json
def func(args):
    print(args["outputSize"])
with open("A:\mouseElephantCatDog\ModelBuilder/test.json") as f: 
    data = json.load(f)
    args = data[1]["args"]
print(args)
func(args)
# {'inputSize': [], 'outputSize': [32, 5]}
# **args -> inputSize = [], outputSize = [32, 5]

