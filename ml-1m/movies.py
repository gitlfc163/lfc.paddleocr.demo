# 读取电影信息
# 1、统计电影ID信息
# 2、统计电影名字的单词，并给每个单词一个数字序号
# 3、统计电影类别的单词，并给每个单词一个数字序号
# 4、电影类别和电影名称定长填充，并保存所有电影数据到字典中
def get_movie_info(path):
    # 打开文件，编码方式选择ISO-8859-1，读取所有数据到data中 
    with open(path, 'r', encoding="ISO-8859-1") as f:
        data = f.readlines()
    # 建立三个字典，分别用户存放电影所有信息，电影的名字信息、类别信息
    movie_info, movie_titles, movie_cat = {}, {}, {}
    # 对电影名字、类别中不同的单词计数
    t_count, c_count = 1, 1
    # 初始化电影名字和种类的列表
    titles = []
    cats = []
    count_tit = {}
    
    # 按行读取数据并处理
    for item in data:
        item = item.strip().split("::")
        v_id = item[0]
        v_title = item[1][:-7]
        cats = item[2].split('|')
        v_year = item[1][-5:-1]

        titles = v_title.split()

        # 2、统计电影名字的单词，并给每个单词一个序号，放在movie_titles中
        for t in titles:
            if t not in movie_titles:
                movie_titles[t] = t_count
                t_count += 1

        # 3、统计电影类别单词，并给每个单词一个序号，放在movie_cat中
        for cat in cats:
            if cat not in movie_cat:
                movie_cat[cat] = c_count
                c_count += 1
        
        # 4、补0使电影名称对应的列表长度为15
        v_tit = [movie_titles[k] for k in titles]
        while len(v_tit)<15:
            v_tit.append(0)
        # 4、补0使电影种类对应的列表长度为6
        v_cat = [movie_cat[k] for k in cats]
        while len(v_cat)<6:
            v_cat.append(0)

        # 4、保存电影数据到movie_info中
        movie_info[v_id] = {'mov_id': int(v_id),
                            'title': v_tit,
                            'category': v_cat,
                            'years': int(v_year)}
    return movie_info, movie_cat, movie_titles


# 5、测试
movie_info_path = "./work/ml-1m/movies.dat"
movie_info, movie_cat, movie_titles = get_movie_info(movie_info_path)
print("电影数量：", len(movie_info))
ID = 1
print("原始的电影ID为 {} 的数据是：".format(ID), data[ID-1])
print("电影ID为 {} 的转换后数据是：".format(ID), movie_info[str(ID)])
print("电影种类对应序号：'Animation':{} 'Children's':{} 'Comedy':{}".format(movie_cat['Animation'], 
                                                                   movie_cat["Children's"], 
                                                                   movie_cat['Comedy']))
print("电影名称对应序号：'The':{} 'Story':{} ".format(movie_titles['The'], movie_titles['Story']))