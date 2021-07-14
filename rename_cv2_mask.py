# 3rename_cv2_mask.py
import os
import shutil

# 将transform_json 下各个文件夹对应的label.png，以文件夹的名字命名后，复制到train_data/cv2_mask
target_dir_name = 'train_data/labelme_json'
source_dir_name = 'transform_json'
for dir in os.listdir(source_dir_name):
    print(dir)
    # dir 01_json  --> 01 -->01.png
    new_name = dir.split('_')[0] + '.png'
    # 名称有下划线需要处理
    old_name = os.path.join(source_dir_name, dir, 'label.png')
    # print(old_name)
    # print(new_name)
    # label.png -->01.png  复制到 train_data/cv2_mask
    shutil.copy(old_name, os.path.join('train_data/cv2_mask', new_name))
