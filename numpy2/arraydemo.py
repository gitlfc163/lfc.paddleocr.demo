
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


# ndarray数组的求和方法
def stat_sum():
    arr = np.array([[1,2,3],[4,5,6]])
    print(arr.sum())
    print(np.sum(arr))

# ndarray数组的计算均值
def stat_tmean():
    # 使用arr.mean() 或 np.mean(arr)，二者是等价的
    arr = np.array([[1,2,3],[4,5,6]])
    print(arr.mean())

# ndarray数组的计算方差
def stat_tvar():
    arr = np.array([[1,2,3],[4,5,6]])
    print(arr.var())

# ndarray数组的计算标准差
def stat_tstd():
    arr = np.array([[1,2,3],[4,5,6]])
    print(arr.std())

# ndarray数组的计算最大值
def stat_tmax():
    arr = np.array([[1,2,3],[4,5,6]])    
    print(arr.max())
    # print(np.max(arr))
        
# ndarray数组的计算最小值
def stat_tmin():
    arr = np.array([[1,2,3],[4,5,6]])
    print(arr.min())
    # print(np.min(arr))

# 找出最大元素的索引
def stat_targmax():
    arr = np.array([[1,2,3],[4,5,6]])
    print(arr.argmax())

# 找出最小元素的索引
def stat_targmin():
    arr = np.array([[1,2,3],[4,0,6]])
    print(arr.argmin()) # 4

# 计算所有元素的累加
def stat_tcumsuml():
    arr = np.array([[1,2,3],[4,5,6]])
    print(arr.cumsum()) # [ 1  3  6 10 15 21]

# 计算所有元素的累积
def stat_cumprod():
    arr = np.array([[1,2,3],[4,5,6]])
    print(arr.cumprod()) # [ 1  2  6 24 120  720]

# ndarray数组的沿着第0维求和
def stat_tsum0():
    arr = np.array([[1,2,3],[4,5,6],[3,6,9]])
    # axis=0表示沿着第0维求和，也就是将[1,2,3]求和等于6，[4,5,6]求和等于15，[3, 6, 9]求和等于18
    print(arr.sum(axis=0)) # [ 8 13 18]

# ndarray数组的沿着第1维求平均
def stat_tmean1():
    arr = np.array([[1,2,3],[4,5,6],[3,6,9]])
    # 沿着第1维求平均，也就是将[1,2,3]取平均等于2，[4,5,6]取平均等于5，[3,6,9]取平均等于6
    print(arr.mean(axis=1)) # [2. 5. 6.]

stat_targmin()