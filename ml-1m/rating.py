
# 评分数据
def get_rating_info(path):
    # 打开文件，读取所有行到data中
    with open(path, 'r') as f:
        data = f.readlines()
    # 创建一个字典
    rating_info = {}
    for item in data:
        item = item.strip().split("::")
        # 处理每行数据，分别得到用户ID，电影ID，和评分
        usr_id,movie_id,score = item[0],item[1],item[2]
        if usr_id not in rating_info.keys():
            rating_info[usr_id] = {movie_id:float(score)}
        else:
            rating_info[usr_id][movie_id] = float(score)
    return rating_info

# 获得评分数据
rating_path = "./work/ml-1m/ratings.dat"
rating_info = get_rating_info(rating_path)
print("ID为1的用户一共评价了{}个电影".format(len(rating_info['1'])))