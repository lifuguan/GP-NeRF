import os
import shutil

# 设置文件夹路径和长度限制
dir_path = 'scannet'
max_length = 12

# 遍历文件夹
for foldername in os.listdir(dir_path):
    # 检查文件夹长度是否超过限制
    if len(foldername) > max_length:
        # 如果超过，则删除该文件夹及其内容
        shutil.rmtree(os.path.join(dir_path, foldername))

