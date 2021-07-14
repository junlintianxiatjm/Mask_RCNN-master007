# makedir.py
'''
创建train_data所需的各个目录
'''
import os

dir_list = ['train_data/cv2_mask', 'train_data/pic', 'train_data/json', 'train_data/labelme_json']

for dir in dir_list:
    print(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
