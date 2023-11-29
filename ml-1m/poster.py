
from PIL import Image
import matplotlib.pyplot as plt

# 使用海报图像和不使用海报图像的文件路径不同，处理方式相同
use_poster = True
if use_poster:
    rating_path = "./work/ml-1m/new_rating.txt"
else:
    rating_path = "./work/ml-1m/ratings.dat"

with open(rating_path, 'r') as f:
    data = f.readlines()

# 从新的rating文件中收集所有的电影ID
mov_id_collect = []
for item in data:
    item = item.strip().split("::")
    usr_id,movie_id,score = item[0],item[1],item[2]
    mov_id_collect.append(movie_id)


# 根据电影ID读取图像
poster_path = "./work/ml-1m/posters/"

# 显示mov_id_collect中第几个电影ID的图像
idx = 1

poster = Image.open(poster_path+'mov_id{}.jpg'.format(str(mov_id_collect[idx])))

plt.figure("Image") # 图像窗口名称
plt.imshow(poster)
plt.axis('on') # 关掉坐标轴为 off
plt.title("poster with ID {}".format(mov_id_collect[idx])) # 图像题目
plt.show()