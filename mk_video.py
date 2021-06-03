import numpy as np
import cv2
import os
from tqdm import tqdm

def pic_to_video(root,pic_list, video_name, fps, pic_size):
    """
    图片合成视频
    :root pic_list_root
    :param pic_list: 图片路径列表
    :param video_name: 生成视频的名字
    :param fps: 1s显示多少张图片
    :param pic_size: 图片尺寸
    :return:
    """
    # 'mp4v' 生成mp4格式的视频
    # 'DIVX' 生成avi格式的视频
    if "mp4" in video_name:
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, pic_size)
    elif ".avi" in video_name:
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, pic_size)
    else:
        print("格式错误")
        return None

    for filename in tqdm(pic_list):
        filename = os.path.join(root, filename)
        if os.path.exists(filename):
            video.write(cv2.imread(filename))
    video.release()

if __name__=="__main__":
    path = "results/test"
    file_list = sorted(os.listdir(path))
    pic_to_video(path, file_list, "test.mp4", fps=24,pic_size=(1024,512))
    # print(file_list)
