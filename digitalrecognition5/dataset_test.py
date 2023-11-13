

# 声明数据加载函数，使用训练模式，MnistDataset构建的迭代器每次迭代只返回batch=1的数据
import paddle
from mnistdataset import MnistDataset


train_dataset = MnistDataset(mode='train')
# 使用paddle.io.DataLoader 定义DataLoader对象用于加载Python生成器产生的数据，
# DataLoader 返回的是一个批次数据迭代器，并且是异步的；
train_loader = paddle.io.DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
val_dataset = MnistDataset(mode='valid')
val_loader = paddle.io.DataLoader(val_dataset, batch_size=128,drop_last=True)