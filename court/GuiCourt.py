from ast import keyword
import cv2
import normalCourt
import numpy as np

# 窗口大小
windowHeight = 700
windowWidth = 900

# 图片
imgPath = "data/tennisCourt/10.png"

# 存放四个图中的坐标点
imgs = []
points = []

#
ncourt = normalCourt.NormalCourt()


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


def showImage(image, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, 900, 720)
    cv2.imshow(name, image)
    keycode = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if keycode & 0xff == 27:
        print("Running end！")
        exit(0)
    else:
        return

def showExample(windowName="example"):
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(windowName, windowWidth , windowHeight)
    img = cv2.imread("./court/example.png")
    cv2.imshow(windowName,img)
    # cv2.waitKey(0)

def showVideoResult(videoPath,frame_i,lines):
    video = cv2.VideoCapture(videoPath)
    fps, length, width, height = get_video_properties(video)
    cv2.namedWindow("video Result", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("video Result", windowWidth , windowHeight)
    print('Video  FPS:{} Length:{} Width:{} Height:{} '.format(fps,length,width,height))
    i = 1
    while True:
        ret, frame = video.read()   # 读取视频
        if ret:
            if i >frame_i:
                for i in range(0, len(lines), 4):   # 绘制球场线
                    x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
                    cv2.line(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255), 3)
            cv2.imshow("video Result", frame)
            i += 1
            key = cv2.waitKey(15)
            if key == 27:
                cv2.destroyAllWindows()
        else:
            break

def setup():
    print("Setup Start...")
    configuration = ncourt.court_conf[1]
    matrix, _ = cv2.findHomography(np.float32(configuration), np.float32(points), method=0)
    print("matrix:",matrix)
    inv_matrix = cv2.invert(matrix)[1]
    # 获取标准球场的所有直线参数
    p = np.array(ncourt.get_important_lines(), dtype=np.float32).reshape((-1, 1, 2))
    # 将标准球场的直线 透视 到当前图像的视角下
    lines = cv2.perspectiveTransform(p, matrix).reshape(-1)
    img = imgs[0].copy()
    for i in range(0, len(lines), 4):   # 绘制球场线
        x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
        cv2.line(img, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255), 3)
    showImage(img, "result")
    return lines, matrix, inv_matrix


def drawPoints(imgPath=None,img=None, windowName="Left click Mark, right click Undo, Enter Draw the court"):
    count = 0
    def onMouse(event, x, y, flags, param):
        nonlocal count 
        if event == cv2.EVENT_MOUSEMOVE:
            img = imgs[-1].copy()
            cv2.line(img,(0,y),(img.shape[1],y),(5,255,238),2)
            cv2.line(img,(x,0),(x,img.shape[0]),(5,255,238),2)
            cv2.imshow(windowName,img)
            
        if event == cv2.EVENT_LBUTTONDOWN:
            count += 1
            img = imgs[-1].copy()
            cv2.circle(img, (x,y), 5, (0,0,255), -1)
            cv2.putText(img,str(count),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=1, thickness=3)
            points.append([x,y])
            imgs.append(img)
            print("Drawing Success! Now Points:", points)
            cv2.imshow(windowName,imgs[-1])
        elif event == cv2.EVENT_RBUTTONDOWN:
            count -= 1
            points.pop()
            imgs.pop()
            print("CTRL-Z Success! Now Points:",points)
            cv2.imshow(windowName,imgs[-1])
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(windowName, windowWidth , windowHeight)
    showExample()
    cv2.setMouseCallback(windowName, onMouse)
    if imgPath is not None:
        img = cv2.imread(imgPath)
    if img is not None:
        img = img
    if imgPath is None and img is None:
        return 0
    imgs.append(img)
    cv2.imshow(windowName,imgs[-1])
    key = cv2.waitKey(0)
    if key == 27:   # Esc
        cv2.destroyAllWindows()
    elif key == 13: # Enter
        lines, matrix, inv_matrix = setup()
    return lines, matrix, inv_matrix

def drwaPointsOnVideo(videoPath, windowName="Left click Mark, right click Undo, Enter Draw the court"):
    video = cv2.VideoCapture(videoPath)
    fps, length, width, height = get_video_properties(video)
    print('Video  FPS:{} Length:{} Width:{} Height:{} '.format(fps,length,width,height))
    frame_i = 1
    img = None
    while True:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
        ret, frame = video.read()   # 读取视频
        if ret:
            windowNameChoose = "Choose Frame to Start![Press Enter to start marking points and press space to jump to the next second]"
            cv2.namedWindow(windowNameChoose, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(windowNameChoose, windowWidth , windowHeight)
            img = frame.copy()
            cv2.putText(img, "Press Enter to start marking points.Press space to jump to the next second...",(100,100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color=(0,0,255), thickness=3)
            cv2.imshow(windowNameChoose, img)
            keycode = cv2.waitKey(0)
            if keycode == 27:   # Esc
                cv2.destroyAllWindows()
                return 0
            elif keycode == 13: # Enter
                lines, matrix, inv_matrix = drawPoints(img=frame)
                cv2.destroyWindow(windowNameChoose)
                cv2.destroyWindow(windowName)
                showVideoResult(videoPath,frame_i,lines)
                return 0
            elif keycode == ord(' '):
                frame_i += int(fps)
        else:
            break

if __name__ == "__main__":
    # drawPoints(imgPath=imgPath)
    drwaPointsOnVideo(videoPath="data/dj/test-13.mp4")