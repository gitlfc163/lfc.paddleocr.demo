import numpy as np

# 用户信息处理
# 1、读取用户数据
# 2、用户的性别F、M是字母数据，这里需要转换成数字表示
# 3、把用户数据的字符串类型的数据转成数字类型，并存储到字典中
def get_usr_info(path):
    # 1、读取用户数据
    usr_file = path # './work/ml-1m/users.dat'
    # 打开文件，获取所有行到data中
    with open(usr_file, 'r') as f:
        data = f.readlines()

    # 2、用户的性别F、M是字母数据，这里需要转换成数字表示
    def gender2num(gender):
        if gender == 'F':
            return 0
        else:
            return 1

    # 3、把用户数据的字符串类型的数据转成数字类型，并存储到字典中
    use_info = {}
    max_usr_id = 0
    # 按行索引数据
    for item in data:
        # 去除每一行中和数据无关的部分
        item = item.strip().split("::")
        usr_id = item[0]
        # 将字符数据转成数字并保存在字典中
        use_info[usr_id] = {'usr_id': int(usr_id),
                            'gender': gender2num(item[1]),
                            'age': int(item[2]),
                            'job': int(item[3])}
        max_usr_id = max(max_usr_id, int(usr_id))

    return use_info, max_usr_id

# 4、测试
usr_file = "./work/ml-1m/users.dat"
usr_info, max_usr_id = get_usr_info(usr_file)
print("用户数量:", len(usr_info))
print("最大用户ID:", max_usr_id)
print("第1个用户的信息是：", usr_info['1'])