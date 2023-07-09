import os
import numpy as np
import json
import torch.nn as nn
import torch
import queue
from torchsummary import summary
from blockDictionary import callBlock
class modelBuilderPreparer():
	def __init__(self, file):
		self.data = json.load(file)
		self.dataLength = len(self.data)
		self.topologicalOrder = self.toTopological()
	def toTopological(self):
		topologicalArray = []
		inDeg = np.zeros(self.dataLength)
		topologicalQueue = queue.Queue(self.dataLength)
		for i in range(self.dataLength):
			inDeg[i] = len(self.data[i]["parent"])
			if(inDeg[i] == 0):
				topologicalQueue.put(i)
				topologicalArray.append(i)
		while topologicalQueue.empty() == False:
			index = topologicalQueue.get()
			for child in self.data[index]["child"]:
				inDeg[child] -= 1
				if(inDeg[child] == 0):
					topologicalQueue.put(child)
					topologicalArray.append(child)
		return topologicalArray
	def getAttr(self):
		return [self.data, self.dataLength, self.topologicalOrder]
class model(nn.Module):
	def __init__(self, data, dataLength, topologicalOrder): 
		super().__init__()
		self.data = data
		self.dataLength = dataLength
		self.topologicalOrder = topologicalOrder
		self.layersArr = [0] * self.dataLength
		for idx in self.topologicalOrder:
			self.layersArr[idx] = callBlock[self.data[idx]["blockName"]](self.layersArr, self.data[idx]["parent"], idx, *self.data[idx]["args"].values())
		self.layers = nn.ModuleList(self.layersArr)
		self.layers.cuda() # cuda :)
	def initWeight(self):
		raise NotImplementedError #TODO
	def forward(self, x): #problem: multi-input, which goes to which?
		xArray = [0] * self.dataLength
		for idx in self.topologicalOrder:
			if(len(self.data[idx]["parent"]) == 0):
				xArray[idx] = self.layers[idx](x)
			elif(len(self.data[idx]["parent"]) == 1): # single input
				xArray[idx] = self.layers[idx](xArray[self.data[idx]["parent"][0]]) #single input
			else:
				xArray[idx] = self.layers[idx]([xArray[j] for j in self.data[idx]["parent"]]) # multiple input
			#print([i.shape if type(i) == torch.Tensor else 0 for i in xArray])
		return xArray[self.topologicalOrder[-1]]
def modelBuilder(path): # should be able to choose output method
	with open(os.path.join(path)) as f:
		preparer = modelBuilderPreparer(f)      
		torchModel = model(*preparer.getAttr()).cuda()
		dummyInput = torch.rand([32,2]).cuda()
		print(torchModel)
		summary(torchModel,dummyInput)
		torch.onnx.export(torchModel,dummyInput,"model.onnx")
path = os.path.join('./test.json')
modelBuilder(path)
