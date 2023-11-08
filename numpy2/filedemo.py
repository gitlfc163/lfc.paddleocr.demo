
# NumPy保存和导入文件
import numpy as np

# 读取文件
def file_read():
    data=np.fromfile("./work/housing.data", sep=' ')
    print("读取文件内容：\n{}".format(data))

# 文件保存和加载
def file_test():
    data=np.array([1,2,3,4,5])
    # 保存内容到文件
    np.save("./data/test_save.npy",data)
    print("保存文件内容：{},到./data/test_save.npy".format(data))
    
    # 加载文件内容
    data2=np.load("./data/test_save.npy")
    print("从/data/test_save.npy加载文件,{}".format(data2))

    # 比较两个数组是否相等
    check = (data == data2).all()
    print("比较两个数组是否相等\n{}".format(check))

    # 追加内容到文件
    # data3=np.array([6,7,8,9,10])
    # np.save("./data/test_save.npy",data3,allow_pickle=True)

    # 加载文件内容
    # data4=np.load("./data/test_save.npy")
    # print("追加内容后从/data/test_save.npy加载文件,{}".format(data4))

file_test()