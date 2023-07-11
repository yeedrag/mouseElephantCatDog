from blocks import blockClass # 應該不用每個py都import一次吧很蠢耶 
#IDK https://stackoverflow.com/questions/1057431/how-to-load-all-modules-in-a-folder
# 你看下這個 不然就是全部import 哈哈 應該沒什麼差別
# 其實也要寫很多行 一個一個弄吧 哈哈
from blocks import convolutionLayers
from blocks import linear
from blocks import nonLinearActivation
from blocks import utilLayers
from blocks import recurrentLayers
from blocks import normalization
from blocks import dropOut

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

