
# 异步训练
# 定义网络结构，同上一节所使用的网络结构
import paddle.nn.functional as F
import paddle
from mnistdataset import MnistDataset

class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        # 定义一层全连接层，输出维度是1
        self.fc = paddle.nn.Linear(in_features=784, out_features=1)

    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs

def train(model):
    model = MNIST()
    model.train()
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    EPOCH_NUM = 10

    # 声明数据加载函数，使用MnistDataset数据集
    train_dataset = MnistDataset(mode='train')
    # 使用paddle.io.DataLoader 定义DataLoader对象用于加载Python生成器产生的数据，
    # DataLoader 返回的是一个批次数据迭代器，并且是异步的；
    data_loader = paddle.io.DataLoader(train_dataset, batch_size=100, shuffle=True)

    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(data_loader()):
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels).astype('float32')
            
            #前向计算的过程  
            predicts = model(images)

            #计算损失，取一个批次样本损失的平均值
            loss = F.square_error_cost(predicts, labels)
            avg_loss = paddle.mean(loss)       
            
            #每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    #保存模型参数
    paddle.save(model.state_dict(), './model/mnist')

#创建模型
# model = MNIST()
#启动训练过程
# train(model)