# 测试           
import paddle
from mnist import MNIST,train,train_multi_gpu


# 1、单GPU训练
# 创建模型    
model = MNIST()
# 启动训练过程
train(model)

# 2、分布式训练--单机多卡程序
# paddle.set_device('gpu')
# #创建模型    
# model = MNIST()
# #启动训练过程
# train_multi_gpu(model)