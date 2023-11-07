
# 模型设计：模型设计是深度学习模型关键要素之一，也称为网络结构设计，相当于模型的假设空间，即实现模型“前向计算”（从输入到输出）的过程。
# 导入需要用到的package
import numpy as np
import json

# 定义网络结构
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        # np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
        
    # 前向计算
    # 从特征和参数计算输出值的过程
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    
    # 损失函数
    # 对一个样本计算损失函数值的实现
    def loss(self, z, y):
        error = z - y
        cost = error * error
        cost = np.mean(cost)
        return cost
    
    # 计算梯度
    def gradient(self, x, y):
        """
        计算梯度

        参数:
            x: 输入特征向量
            y: 目标值

        返回:
            gradient_w: 权重的梯度
            gradient_b: 偏置的梯度
        """
        z = self.forward(x)
        gradient_w = (z-y)*x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, eta = 0.01):
        """
        更新模型参数

        参数:
            gradient_w: 权重的梯度
            gradient_b: 偏置的梯度
            eta: 学习率（可选，默认值为0.01）
        """
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
        
    def train(self, x, y, iterations=100, eta=0.01):
        """
        训练模型

        参数:
        x: 输入数据
        y: 目标值
        iterations: 迭代次数，默认为100
        eta: 学习率，默认为0.01
        """

        losses = []  # 损失值列表
        for i in range(iterations):
            z = self.forward(x)  # 前向传播
            L = self.loss(z, y)  # 计算损失
            gradient_w, gradient_b = self.gradient(x, y)  # 计算梯度
            self.update(gradient_w, gradient_b, eta)  # 更新参数
            losses.append(L)  # 记录损失值
            if (i+1) % 10 == 0:  # 每10次迭代打印一次迭代次数和损失值
                print('iter {}, loss {}'.format(i, L))
        return losses  # 返回损失值列表
    
    def train(self, training_data, num_epochs, batch_size=10, eta=0.01):
        """
        训练模型

        参数:
        training_data: 训练数据集
        num_epochs: 迭代次数
        batch_size: 每个迭代周期中包含的数据数量，默认为10
        eta: 学习率，默认为0.01

        返回值:
        losses: 每个迭代周期的损失值列表
        """
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epochs):
            # 在每轮迭代开始之前，将训练数据的顺序随机打乱
            # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
            mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
                #print(self.w.shape)
                #print(self.b)
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                                 format(epoch_id, iter_id, loss))
        
        return losses