# 测试           

# 1、计算模型的分类准确率
# from mnist import MNIST,train
# 创建模型    
# model = MNIST()
# 启动训练过程
# train(model)

# 2、检查模型训练过程，识别潜在训练问题
# from mnistf import MNIST,train
# # 创建模型    
# model = MNIST()
# # 启动训练过程
# train(model)
# print("Model has been saved.")


# 3、加入校验或测试，更好评价模型效果(需先生成模型)
# from mnistf import MNIST
# from evaluation import evaluation

# # 创建模型    
# model = MNIST()
# # 启动测试过程
# evaluation(model)


# 4、加入正则化项，避免模型过拟合
# from mnistf import MNIST,train_regul
# model = MNIST()
# train_regul(model)


# 5、可视化分析
# 5.1、使用Matplotlib库绘制损失随训练下降的曲线图
# from mnistf import MNIST,train_matplotlib
# # 引入matplotlib库
# import matplotlib.pyplot as plt
# model = MNIST()
# iters, losses = train_matplotlib(model)

# # 画出训练过程中Loss的变化曲线
# plt.figure()
# plt.title("train loss", fontsize=24)
# plt.xlabel("iter", fontsize=14)
# plt.ylabel("loss", fontsize=14)
# plt.plot(iters, losses,color='red',label='train loss') 
# plt.grid()
# plt.show()


# 5、可视化分析
# 5.2、使用VisualDL可视化分析
from mnistf import MNIST,train_visualdl
model = MNIST()
train_visualdl(model)
