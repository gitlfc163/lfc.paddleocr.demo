
# 加载相关库
import os
import random
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
import numpy as np
from PIL import Image
import gzip
import json

from mnist import MNIST

# # 创建模型    
# model = MNIST()
# # 启动训练过程
# train(model)

# 读取一张本地的样例图片，转变成模型输入的格式
def load_image(img_path):
    # 从img_path中读取图像，并转为灰度图
    im = Image.open(img_path).convert('L')
    # im = im.resize((28, 28), Image.ANTIALIAS) # 报错“AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'”
    im = im.resize((28, 28), Image.LANCZOS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    # 图像归一化
    im = 1.0 - im / 255.
    return im

# 定义预测过程
model = MNIST()
params_file_path = './model/mnist.pdparams'
img_path = './work/example_0.jpg'
# 加载模型参数
param_dict = paddle.load(params_file_path)
model.load_dict(param_dict)
# 灌入数据
model.eval()
tensor_img = load_image(img_path)
# 模型反馈10个分类标签的对应概率
results = model(paddle.to_tensor(tensor_img))
# 取概率最大的标签作为预测输出
lab = np.argsort(results.numpy())
print("本次预测的数字是: ", lab[0][-1])