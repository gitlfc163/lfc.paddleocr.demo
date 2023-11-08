

# 随机数np.random
import numpy as np

# 生成均匀分布随机数
def random_rand(seed):
    # 设置随机数种子
    np.random.seed(seed)
    # 生成均匀分布随机数，随机数取值范围在[0, 1)之间
    arr=np.random.rand(3,3)
    print(arr)

# 生成正态分布随机数
def random_randn(seed):
    # 设置随机数种子
    np.random.seed(seed)
    # 生成正态分布随机数，随机数取值范围在[-1, 1)之间
    arr=np.random.randn(3,3)
    print(arr)

# 生成正态分布随机数，指定均值loc和方差scale
def random_normal(seed):
    # 设置随机数种子
    np.random.seed(seed)
    # 生成正态分布随机数，随机数取值范围在[loc-scale, loc+scale)之间
    arr=np.random.normal(loc=0, scale=1, size=(3,3))
    print(arr)


# 随机打乱1维ndarray数组顺序
def random_shuffle():
    # 生成1维ndarray数组
    arr=np.arange(0,30)
    print(arr)
    # 随机打乱数组顺序
    np.random.shuffle(arr)
    print(arr)

# 随机打乱2维ndarray数组顺序
def random_shuffle2(stop):
    # 生成一维数组
    arr = np.arange(0, stop)
    # 将一维数组转化成2维数组
    arr = arr.reshape(5, 6)
    print('before random shuffle: \n{}'.format(arr))
    # 随机打乱数组顺序
    np.random.shuffle(arr)
    print('after random shuffle: \n{}'.format(arr))


# 随机选取元素
def random_choice():
    # 生成1维数组
    arr = np.arange(0, 30)
    print('生成的1维数组: \n{}'.format(arr))
    # 随机选取数组中的一个元素
    print('随机选取数组中的一个元素: \n{}'.format(np.random.choice(arr)))
    # 随机选取数组中的多个元素
    print('随机选取数组中的多个元素: \n{}'.format(np.random.choice(arr, size=3)))




# 以一维数组的形式返回方阵的对角线（或非对角线）元素
def diagonal(seed):
    # 设置随机数种子
    np.random.seed(seed)
    # 生成方阵
    arr = np.random.randn(3, 3)
    print('生成的方阵: \n{}'.format(arr))
    # 以一维数组的形式返回方阵的对角线（或非对角线）元素
    e = np.diag(arr)
    print('方阵的对角线元素: \n{}'.format(e))


# 矩阵乘法
def matrix_multiply(seed):
    # 设置随机数种子
    np.random.seed(seed)
    # 生成方阵
    arr1 = np.random.randn(3, 3)
    arr2 = np.random.randn(3, 3)
    print('arr1: \n{}'.format(arr1))
    print('arr2: \n{}'.format(arr2))
    print('arr1 * arr2: \n{}'.format(arr1 * arr2))


# 计算对角线元素的和
def diagonal_trace(seed):
    # 设置随机数种子
    np.random.seed(seed)
    # 生成方阵
    arr = np.random.randn(3, 3)
    print('生成的方阵: \n{}'.format(arr))
    # 计算对角线元素的和
    print('方阵的对角线元素和: \n{}'.format(np.trace(arr)))


# det，计算行列式
def det_test(seed):
    # 设置随机数种子
    np.random.seed(seed)
    # 生成方阵
    arr = np.random.randn(3, 3)
    print('生成的方阵: \n{}'.format(arr))
    # 计算行列式
    print('方阵的行列式: \n{}'.format(np.linalg.det(arr)))

# 计算方阵的特征值和特征向量
def eig_test(seed):
    # 设置随机数种子
    np.random.seed(seed)
    # 生成方阵
    arr = np.random.randn(3, 3)
    print('生成的方阵: \n{}'.format(arr))
    # 计算方阵的特征值和特征向量
    print('方阵的特征值和特征向量: \n{}'.format(np.linalg.eig(arr)))

# inv，计算方阵的逆
def inv_test(seed):
    # 设置随机数种子
    np.random.seed(seed)
    # 生成方阵
    arr = np.random.randn(3, 3)
    print('生成的方阵: \n{}'.format(arr))
    # 计算方阵的逆
    print('方阵的逆: \n{}'.format(np.linalg.inv(arr)))

inv_test(30)