

# 本节主要探讨在手写数字识别任务中，使得损失达到最小的参数取值

# 加载相关库
import os
import random
from paddle.nn import Conv2D, MaxPool2D, Linear
import numpy as np
import gzip
import json

# 定义数据集读取器
def load_data(mode='train'):
    print("__file__{}".format(os.path.dirname(__file__)))
    datafile = './work/mnist.json.gz'
    # 或者
    # datafile = os.path.join(os.path.dirname(__file__), 'work\mnist.json.gz')
    # if not os.path.exists(datafile):
    #     print("not exists datafile,{}".format(datafile))

    print("load data from {}".format(datafile))
    data=json.load(gzip.open(datafile))
    # 读取数据集中的训练集，验证集和测试集
    train_set, val_set, test_set = data
    
    # 数据集相关参数，图片高度和宽度
    IMG_ROWS = 28
    IMG_COLS = 28
    # 根据不同的模式读取不同的图片和标签
    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'eval':
        imgs = test_set[0]
        labels = test_set[1]
    
    # 获取所有图像的数量
    imgs_length = len(imgs)
    # 验证数据集的数量是否正确
    assert imgs_length == len(labels), \
           "length of train_imgs({}) should be equal to length of train_labels({})".format(imgs_length, len(labels))
    
    index_list = list(range(imgs_length))      
                                                                           
    # 读入数据时用到的batchsize
    BATCHSIZE = 100 
    
    def data_generator():
       if mode == 'train':
           random.shuffle(index_list)
       imgs_list=[]
       labels_list=[]
       for i in index_list:
           # 读取图像和标签，并转换其尺寸和类型
           img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype("float32")
           label=np.reshape(labels[i], [1]).astype("int64")
           imgs_list.append(img)
           labels_list.append(label)
           # 如果当前数据已经读取完一个batchsize的数据，就返回这个batchsize的数据
           if len(imgs_list) == BATCHSIZE:
               yield np.array(imgs_list), np.array(labels_list)
               # 清空数据缓存列表
               imgs_list=[]
               labels_list=[]
        
        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据的数目可以用来构建一个大小为len(imgs_list)的batch
       if len(imgs_list) > 0:
           yield np.array(imgs_list), np.array(labels_list)   
        
    return data_generator  
 