# lfc.paddleocr.demo

## 简介

模型加载及恢复训练
在快速入门中，我们已经介绍了将训练好的模型保存到磁盘文件的方法。应用程序可以随时加载模型，完成预测任务。但是在日常训练工作中我们会遇到一些突发情况，导致训练过程主动或被动的中断。如果训练一个模型需要花费几天的训练时间，中断后从初始状态重新训练是不可接受的。

万幸的是，飞桨支持从上一次保存状态开始训练，只要我们随时保存训练过程中的模型状态，就不用从初始状态重新训练。