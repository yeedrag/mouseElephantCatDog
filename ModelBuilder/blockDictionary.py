from ModelBuilder.blocks import blockClass # 應該不用每個py都import一次吧很蠢耶 
#IDK https://stackoverflow.com/questions/1057431/how-to-load-all-modules-in-a-folder
# 你看下這個 不然就是全部import 哈哈 應該沒什麼差別
# 其實也要寫很多行 一個一個弄吧 哈哈
from ModelBuilder.blocks import convolutionLayers
from ModelBuilder.blocks import linear
from ModelBuilder.blocks import nonLinearActivation
from ModelBuilder.blocks import utilLayers
from ModelBuilder.blocks import recurrentLayers
from ModelBuilder.blocks import normalization
from ModelBuilder.blocks import dropOut

callBlock = {
    "Linear": linear.Linear, 
    "Activation": nonLinearActivation.Activation,
    "Input": utilLayers.Input,
    "Concat": utilLayers.Concat,
    "Conv": convolutionLayers.Conv,
    "Pooling": convolutionLayers.Pooling,
    "Flatten": utilLayers.Flatten,
    "Dropout": dropOut.Dropout
}

