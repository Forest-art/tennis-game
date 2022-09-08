
import os
import random
import time


from tqdm import tqdm
from courtDetector import CourtDetector
import cv2
import tqdm
import numpy as np
from matplotlib import pyplot as plt



def showImage(image, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, 900, 720)
    cv2.imshow(name, image)
    keycode = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if keycode & 0xff == 27:
        print("Running end！")
        exit(0)


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

def test(videoPath,verbose=0, args=None):
    court_detector = CourtDetector(verbose=verbose,args=args)  
    video = cv2.VideoCapture(videopath)
    print('Video FPS ', video.get(cv2.CAP_PROP_FPS))
    fps, length, width, height = get_video_properties(video)

    # frame counter
    frame_i = 0
    courtlines = [] # 存放每一帧的球场直线坐标
    while True:
        ret, frame = video.read()   # 读取视频
        # print("frame.shape",frame.shape)
        frame_i += 1    
        if ret:
            if showtime:
                startTime = time.time()
            # y,x = frame.shape[0:2]
            # frame = cv2.resize(frame,(int(x/2),int(y/2)))
            # print("frame.resize.shape",frame.shape)
            lines = court_detector.setup(frame)     # 检测球场的直线
            if showtime:
                print("time:", time.time()-startTime)
            if lines is None:
                if len(courtlines):
                    lines = courtlines[-1]
                else:
                    lines = []
            courtlines.append(lines)
            print("\r{}/{}".format(frame_i,length),end="")

            if showResult:
                for i in range(0, len(lines), 4):   # 绘制球场线
                    x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
                    cv2.line(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255), 3)
                showImage(frame,"result")
        else:
            break

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')	# 视频编解码器 mp4
    # 文件输出位置
    out = cv2.VideoWriter(videopath.split('/')[-1], fourcc, fps, (width, height))
    # 存储视频
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret,frame = video.read()
    for i in tqdm.tqdm(range(len(courtlines))):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret,frame = video.read()
        if not ret:
            if i==0:
                continue
            frame = last_frame
        last_frame = frame.copy()
        lines = courtlines[i]
        if len(lines)>0:
            for i in range(0, len(lines), 4):   # 绘制球场线
                x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
                cv2.line(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255), 3)
        out.write(frame)
    video.release()
    print('Input Video Released')
    out.release()
    print("Output Video Saved in ",videopath.split('/')[-1])


def testwithDropPoint(videoPath):
    court_detector = CourtDetector()  
    video = cv2.VideoCapture(videopath)
    print('Video FPS ', video.get(cv2.CAP_PROP_FPS))
    fps, length, width, height = get_video_properties(video)

    # frame counter
    frame_i = 0
    courtlines = []
    normalCourtFrames = []    # 获取标准球场
    originPoints = []     # 原图落点list
    dropPoints = [] # 球场图落点list
    while True:
        ret, frame = video.read()   # 读取视频
        if ret:
            lines = court_detector.setup(frame)     # 检测球场的直线
            if lines is None:
                if len(courtlines):
                    lines = courtlines[-1]
                else:
                    lines = []
            courtlines.append(lines)
            print("\r{}/{}".format(frame_i,length),end="")
            if showResult:
                for i in range(0, len(lines), 4):   # 绘制球场线
                    x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
                    cv2.line(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255), 3)
                showImage(frame,"result")

             # 随机生成落点
            if frame_i % 60 == 0:
                points = [(random.randint(int(width/4), int(width*2/3)), random.randint(int(height/4),int(height*3/4)))]
                originPoints.append(points)
                new_points = court_detector.get_drop_point(points=points)   # 投影变换
                dropPoints.append(new_points)
            else:
                if frame_i == 0:
                    originPoints.append([])
                else:
                    originPoints.append(originPoints[-1])
                dropPoints.append([])
            frame_i += 1
            
        else:
            break

    print("Video saved in ",videopath.split('/')[-1])

    normal_court_image = cv2.cvtColor(court_detector.normal_court.court.copy(),cv2.COLOR_GRAY2BGR)
    artio =  height/ normal_court_image.shape[0]    # 以高度为准，按比例缩放标准球场图
    normalwidth = int(artio * normal_court_image.shape[1])   # 球场的宽
    # 文件输出位置
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')	# 视频编解码器 mp4
    out = cv2.VideoWriter(videopath.split('/')[-1], fourcc, fps, (width+normalwidth, height))
    # 存储视频
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret,frame1 = video.read()
    
    for i in tqdm.tqdm(range(len(courtlines))):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret,frame1 = video.read()
        if not ret:
            if i==0:
                continue
            frame1 = last_frame
        last_frame = frame1.copy()

        lines = courtlines[i]
        if len(lines)>0:
            for i in range(0, len(lines), 4):   # 绘制球场线
                x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
                cv2.line(frame1, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255), 3)
        points = originPoints[i]
        # print(points)
        if len(points) != 0:
            cv2.circle(frame1, (points[0][0],points[0][1]), 10, (255,0,0), -1)
        new_points = dropPoints[i]
        # print(new_points)
        if len(new_points) != 0:
            cv2.circle(normal_court_image, (int(new_points[0][0]),int(new_points[0][1])), 50, (255,(frame_i%60)*30,0), -1)

        frame1 = cv2.resize(frame1,(width,height))   # 原图
        frame2 = cv2.resize(normal_court_image.copy(),(normalwidth,height))  # 标准球场图
        img = np.hstack((frame1, frame2))   # 拼接 原图+标准球场图
        out.write(img)

    video.release()
    print('Video Released')
    out.release()


# 每一帧检测完是否展示结果
showResult = True
showtime = True

if __name__=="__main__":
    # videopath = "data/video/3_Trim.mp4"
    videopath = "data/video/1.mp4"
    # videopath = "data/video/4_Trim.mp4"
    # videopath = "data/video/trim30.mp4"
    # videopath = "data/video/trim30_Trim.mp4"
    # videopath = "data/video/20220826-1.mp4"
    # videopath = "data/video/20220826-2.mp4"
    # videopath = "data/dj/test-11.mp4" 
    # videopath = "data/dj/dji_mimo_20220827_152232_8_1661587832492_video.mp4"  
    # videopath = "data/dj/dji_mimo_20220827_155216_13_1661591185129_video.mp4"  
    # videopath = "data/dj/test-6.mp4"  
    # videopath = "data/dj/test-7.mp4"  
    # videopath = "data/dj/test-8.mp4"  
    # videopath = "data/dj/test-10.mp4"  
    # videopath = "data/dj/test-11.mp4"  
    # videopath = "data/dj/test-12.mp4"  
    # videopath = "data/dj/test-13.mp4"  
    args = {
        "threshold": 150,    # _threshold 控制图像中提取的白色像素的起始范围
        "minLineLength": 200, # _detect_lines HoughLinesP 可以检测出的最小线段长度
        "maxLineGap": 40, # _detect_lines HoughLinesP 统一方向上两条线段定为一条线段的最大允许间隔
        "angle": 15,     # _classify_lines 曲分垂直线和水平线的角度
        "filterTop":0.55,    # _classify_lines 过滤图片 小于height*0.5 的直线
        "filterBottom":0.9,     # _classify_lines 过滤图片 大于height*0.8 的直线
        "minHorizontal": 30,    # _merge_lines 合并水平直线时，小于minHorizontal的进行合并
        "minVertical": 30,    # _merge_lines 合并垂直直线时，小于minVertical的进行合并
    }
    test(videoPath=videopath,verbose=0, args=args)
    # testwithDropPoint(videoPath=videopath)



