
import os
import random
from tqdm import tqdm
from courtDetector import CourtDetector
import cv2
import tqdm
import numpy as np
from matplotlib import pyplot as plt

def get_video_properties(video):
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # get videos properties
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        length = int(video.get(cv2.cv.CAP_PROP_FRAME_COUNT))
        v_width = int(video.get(cv2.cv.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.cv.CAP_PROP_FRAME_HEIGHT))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        v_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, length, v_width, v_height

if __name__=="__main__":
    # videopath = "data/video/3_Trim2.mp4"
    # videopath = "data/video/4_Trim.mp4"
    videopath = "data/video/trim30.mp4"
    court_detector = CourtDetector()  
    video = cv2.VideoCapture(videopath)
    print('Video FPS ', video.get(cv2.CAP_PROP_FPS))
    fps, length, width, height = get_video_properties(video)

    # frame counter
    frame_i = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')	# 视频编解码器 mp4
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')	# 视频编解码器  avi
    # court_video = cv2.VideoWriter('VideoOutput/court_only.mp4',fourcc, fps, (v_width, v_height))
    frames = []
    # Loop over all frames in the videos
    normalCourtFrames = []    # 获取标准球场

    points = []     # 落点list
    while True:
        ret, frame = video.read()   # 读取视频
        frame_i += 1    
        if ret:
            lines = court_detector.setup(frame)     # 检测球场的直线
            print("\r{}/{}".format(frame_i,length),end="")
            for i in range(0, len(lines), 4):   # 绘制球场线
                x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
                cv2.line(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255), 3)
            new_frame = cv2.resize(frame, (width, height))  # 将图片 resize 到指定大小 

            # 获取上一张标准球场图片
            if len(normalCourtFrames) == 0:
                normal_court_image = cv2.cvtColor(court_detector.normal_court.court.copy(),cv2.COLOR_GRAY2RGB)
            else:
                normal_court_image = normalCourtFrames[-1].copy()
            # 随机生成落点
            if frame_i % 60 == 0:
                points = [(random.randint(int(width/4), int(width*2/3)), random.randint(int(height/4),int(height*3/4)))]
                new_points = court_detector.get_drop_point(points=points)   # 投影变换
                # print("Draw drop points of point :", points,"new points:",new_points)

            # 绘制落点
            if len(points) != 0:
                cv2.circle(new_frame, (int(points[0][0]),int(points[0][1])), 10, (255,0,0), -1)
                cv2.circle(normal_court_image, (int(new_points[0][0]),int(new_points[0][1])), 50, (255,(frame_i%60)*30,0), -1)
            # 将绘制后的图片存储在列表中
            normalCourtFrames.append(normal_court_image)
            frames.append(new_frame)
        else:
            break

    video.release()
    print('Video Released')

#
    frame_number = 0
    orig_frame = 0

    
    print("Video saved in ",videopath.split('/')[-1])

    artio =  height/normalCourtFrames[-1].shape[0]    # 以高度为准，按比例缩放标准球场图
    normalwidth = int(artio * normalCourtFrames[-1].shape[1])   # 球场的宽
    # 文件输出位置
    out = cv2.VideoWriter(videopath.split('/')[-1], fourcc, fps, (width+normalwidth, height))
    # 存储视频
    for i in tqdm.tqdm(range(len(frames))):
        orig_frame += 1
        frame1 = cv2.resize(frames[i],(width,height))   # 原图
        frame2 = cv2.resize(normalCourtFrames[i],(normalwidth,height))  # 标准球场图
        img = np.hstack((frame1, frame2))   # 拼接 原图+标准球场图
        out.write(img)
        frame_number += 1
    out.release()



