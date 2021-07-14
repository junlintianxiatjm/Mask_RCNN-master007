# 2transform_json.py
'''
批量转化json文件
'''
import os

path = './'  # path为json文件存放的路径
files = os.listdir(path)

# os.system("activate labelme")
# for file in json_file:
#     # filename = os.path.join(path, file)
#     # print(filename)
#     os.system("labelme_json_to_dataset %s" % (path + '/' + file))


os.system("labelme_json_to_dataset %s" % (path))