
# 异步训练测试           
import paddle
from mnist import MNIST,train

model = MNIST()
# 启动训练过程
train(model)
