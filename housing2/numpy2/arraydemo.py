
import numpy as np

# 案例1：实现a+1的计算
def add_test1():
    a = np.array([1,2,3,4])
    c = a + 1
    print(c)

# 案例2：实现c=a+b的计算
def add_test2():
    a = np.array([1,2,3,4])
    b = np.array([5,6,7,8])
    c = a + b
    print(c)

# 案例3：实现c=a-b的计算
def sub_test1():
    a = np.array([1,2,3,4])
    b = np.array([5,6,7,8])
    c = b - a
    print(c)

# array：创建嵌套序列
def create_test1():
    a = np.array([[1,2,3],[4,5,6]])
    print(a)

# 通过np.arange创建
def create_test2():
    # 创建元素从0到10依次递增2的数组。
    a = np.arange(0,10,2)
    print(a)

# 创建全0的ndarray
def create_test3():
    a = np.zeros((2,3))
    print(a)

# ones：创建指定长度或者形状的全1数组。
def create_test4():
    a = np.ones((2,3))
    print(a)

# 查看ndarray数组的属性
def attr_test1():
    a = np.array([[1,2,3],[4,5,6]])
    print('a, dtype: {}, shape: {}, size: {}, ndim: {}'.format(a.dtype, a.shape, a.size, a.ndim))

# 转化数据类型
def attr_test2():
    a = np.array([[1,2,3],[4,5,6]])
    print('a, dtype: {}, shape: {}, size: {}, ndim: {}'.format(a.dtype, a.shape, a.size, a.ndim))
    b = a.astype(np.float32)
    print('b, dtype: {}, shape: {}'.format(b.dtype, b.shape))

# 改变形状
def attr_test3():
    a = np.array([[1,2,3],[4,5,6]])
    print('a, dtype: {}, shape: {}, size: {}, ndim: {}'.format(a.dtype, a.shape, a.size, a.ndim))
    b = a.reshape(6,1)
    print('b, dtype: {}, shape: {}'.format(b.dtype, b.shape))

# 用标量除以数组的每一个元素
def attr_test4():
    a = np.array([[1,2,3],[4,5,6]])
    b = a / 2
    print(b)

# 数组减去数组
def attr_test5():
    a = np.array([[1,2,3],[4,5,6]])
    b = np.array([[1,2,3],[4,5,6]])
    c = a - b
    print(c)


attr_test5()