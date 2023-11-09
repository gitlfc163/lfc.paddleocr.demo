
# 异步训练测试           
import paddle
from asynmnist import MNIST,train
from mnistdataset import MnistDataset

# # 声明数据加载函数，使用MnistDataset数据集
# train_dataset = MnistDataset(mode='train')
# # 使用paddle.io.DataLoader 定义DataLoader对象用于加载Python生成器产生的数据，
# # DataLoader 返回的是一个批次数据迭代器，并且是异步的；
# data_loader = paddle.io.DataLoader(train_dataset, batch_size=100, shuffle=True)

model = MNIST()
# 启动训练过程
train(model)
