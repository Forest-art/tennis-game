import numpy as np
import cv2
from matplotlib import pyplot as plt
from sympy import Line
from itertools import combinations
try:
    from .normalCourt import NormalCourt
except:
    from normalCourt import NormalCourt

class CourtDetector:
    """
    检测以及跟踪图片中的网球场
    """
    def __init__(self, verbose=0):
        self.verbose = verbose  # 可视化
        self.colour_threshold = 200     # 网球线的白色 阈值 起始值
        self.dist_tau = 3       # 
        self.intensity_threshold = 150   # 光照强度阈值   
        self.normal_court = NormalCourt()    # 标准网球场
        self.v_width = 0    # 输入图像的宽度
        self.v_height = 0   # 输入图像的高度
        self.frame = None   # 当前图像
        self.gray = None    # 当前图像的灰度图
        self.court_warp_matrix = []     # 将标准球场映射到当前图像的透视矩阵列表
        self.game_warp_matrix = []      # 将当前位置坐标映射到标准球场的透视矩阵列表
        self.court_score = 0    # 当前球场的得分
        self.topBaseLine = None    # 上底边
        self.bottomBaseLine = None     # 下底边
        self.net = None     # 中网
        self.leftOuterLine = None    # 左外边线
        self.rightOuterLine = None    # 右外边线
        self.leftInnerLine = None     # 左内边线
        self.rightInnerLine = None    # 右内边线
        self.middleLine = None     # 中线
        self.topInnerLine = None      # 上内边线
        self.bottomInnerLine = None   # 下内边线
        self.best_conf = None       # 最佳匹配的标准球场的四点坐标对应的编号
        self.frame_points = None       # 当前图片中与标准球场最匹配的四点的坐标（标准球场逆向得到）
        self.dist = 3       # 跟踪时,用于判断当前点偏移一定距离后是否可以检测到白色像素
        self.flag = True   # true时进行detect，false进行track
    
    def setup(self,frame, dist=3,retrack_points_num=50,max_offset=6,increment=3):
        """
        检测并跟踪球场
        @parameter:
            dist: 跟踪时,用于判断当前点偏移一定距离后是否可以检测到白色像素
            retrack_points_num: 跟踪时, 检测球场的四条线,每条线采样100个点, 如果最终总点数 < retrack_points_num 则将直线偏移dist后重新计算
            max_offset: 最大偏移距离,超过此偏移距离后任然检测不到则进行重新检测。
            increment: 跟踪失败时对self.dist自增的值。
        @return:
            lines: 但前图像终球场的球场线list
        """
        lines = []
        if self.flag:
            try:
                lines = self.detect(frame=frame,dist=dist,minLineLength=100,maxLineGap=20,verbose=0)
                self.flag = False
            except Exception as e:
                print("No tennis courts detected! <Exception message>:",e)
        else:
            try:
                lines = self.track_court(frame=frame,retrack_points_num=50,max_offset=6,increment=3)
            except Exception as e:
                print("No tennis courts tracked! <Exception message>:",e)
                self.flag = True
        return lines
    
    def get_drop_point(self,points):
        """
        将网球的落点映射到标准网球场noramlCourt上,并返回其坐标
        @parameter:
            points: 真实图片终的网球落点。 【e.g. 1. list: [(1,2),(3,4)]  2. numpy.array [[1 2],[3 4]]】
        @return:
            new_points: 标准球场终的位置坐标
        """
        if isinstance(points,list):
            points = np.array(points)
        points = points.reshape(-1,1,2).astype(np.float32)  # 转变为3维矩阵，float类型
        new_points = cv2.perspectiveTransform(points, self.game_warp_matrix[-1])     # 计算透视点坐标
        new_points = new_points.reshape(-1,2).astype(np.int32)   # 变为二维矩阵,并变为int类型
        return new_points
        
        

    def detect(self, frame, dist=3, minLineLength=300, maxLineGap=60, verbose=0):
        """
        首次检测球场,或者球场位置发生便宜后无法继续跟踪重新检测球场位置
        @parameter:
            frame: 输入图像
            dist: 当前匹配失败后,偏移dist距离进行检测
            minLineLength=100: 检测出的最小线段的长度
            maxLineGap=20: 最大线段间隔,小于此间隔的认为是一条直线
        """
        self.dist = dist   #  球场单次偏移距离
        self.frame_points = None
        self.verbose = verbose  # 可视化
        self.frame = frame      # 当前图像（视频帧）
        self.v_height, self.v_width = frame.shape[:2]   #  图像的高度和宽度

        self.gray = self._threshold(frame)  # 获取图像的灰度图
        # cv2.namedWindow('<func: display_lines_on_frame> gray', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.resizeWindow("<func: display_lines_on_frame> gray", 640, 480)
        # cv2.imshow('<func: display_lines_on_frame> gray', self.gray)

        filtered = self._filter_pixels(self.gray)   # 过滤图像中的非线条类型像素
        # cv2.namedWindow('<func: display_lines_on_frame> filtered', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.resizeWindow("<func: display_lines_on_frame> filtered", 640, 480)
        # cv2.imshow('<func: display_lines_on_frame> filtered', filtered)

        horizontal_lines, vertical_lines = self._detect_lines(filtered,minLineLength=minLineLength,maxLineGap=maxLineGap) # 找到水平线和垂直线
        print("horizontal_lines:",len(horizontal_lines),"vertical_lines:",len(vertical_lines))
        # 找到当前球场与标准球场的透视变换
        court_warp_matrix, game_warp_matrix, self.court_score = self._find_homography(horizontal_lines, vertical_lines)
        self.court_warp_matrix.append(court_warp_matrix)    # 存放当前球场到标准球场的透视矩阵
        self.game_warp_matrix.append(game_warp_matrix)      # 存放标准球场到当前球场的透视矩阵
        return self.find_lines_location()

    def _threshold(self, frame):
        """
        提取图像中的白色像素,返回阈值过滤之后的灰度图
        @parameter:
            frame:待检测图像
        @return:
            gray:阈值过滤之后的灰度图
        """
        # frame = self._enhanceColorHSV(frame=frame)
        # cv2.namedWindow('<func: display_lines_on_frame> _enhanceColor', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.resizeWindow("<func: display_lines_on_frame> _enhanceColor", 640, 480)
        # cv2.imshow('<func: display_lines_on_frame> _enhanceColor', frame)
        # if cv2.waitKey(0) & 0xff == 27:
        #     cv2.destroyAllWindows()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
        # cv2.namedWindow('<func: display_lines_on_frame> gray', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.resizeWindow("<func: display_lines_on_frame> gray", 640, 480)
        # cv2.imshow('<func: display_lines_on_frame> gray', gray)
        # if cv2.waitKey(0) & 0xff == 27:
        #     cv2.destroyAllWindows()
        return gray

    def _enhanceColor(self,frame):
        (B,G,R) = cv2.split(frame)
        imgBlueChannelAvg = np.mean(B)
        imgGreenChannelAvg = np.mean(G)
        imgRedChannelAvg = np.mean(R)
        k = (imgBlueChannelAvg+imgGreenChannelAvg+imgRedChannelAvg)/3
        kb = k/imgBlueChannelAvg
        kg = k/imgGreenChannelAvg
        kr = k/imgRedChannelAvg
        B = cv2.addWeighted(B,kb,0,0,0)
        G = cv2.addWeighted(G,kg,0,0,0)
        R = cv2.addWeighted(R,kr,0,0,0)
        # self.frame = cv2.merge([B,G,R])
        return cv2.merge([B,G,R])

    def _enhanceColorHSV(self,frame):
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # greenMask = cv2.inRange(hsv.copy(),(26,10,30),(97,100,255))
        greenMask = cv2.inRange(hsv.copy(),(97,10,20),(120,100,255))
        cv2.namedWindow('<func: display_lines_on_frame> greenMask', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("<func: display_lines_on_frame> greenMask", 640, 480)
        cv2.imshow('<func: display_lines_on_frame> greenMask', greenMask)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
        hsv[:,:,1] = greenMask
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _filter_pixels(self, gray):
        """
        过滤球场线
        @parameter:
            gray: 检测图像的灰度图
        @return:
            gray: 过滤像素以后的灰度图
        """
        for i in range(self.dist_tau, len(gray) - self.dist_tau):
            for j in range(self.dist_tau, len(gray[0]) - self.dist_tau):
                if gray[i, j] == 0:
                    continue
                # 横向判断左右各 dist_tau的距离像素颜色差值是否超过一定的阈值
                if (gray[i, j] - gray[i + self.dist_tau, j] > self.intensity_threshold and gray[i, j] - gray[i - self.dist_tau, j] > self.intensity_threshold):
                    continue
                # 纵向判断上下各 dist_tau的距离像素颜色差值是否超过一定的阈值
                if (gray[i, j] - gray[i, j + self.dist_tau] > self.intensity_threshold and gray[i, j] - gray[i, j - self.dist_tau] > self.intensity_threshold):
                    continue
                # 赋值为黑色
                gray[i, j] = 0
        return gray

    def _detect_lines(self, gray,minLineLength=100,maxLineGap=20):
        """
        使用Hough检测图像中的所有直线
        @parameter:
            gray: 但前图像的灰度图
            minLineLength=100: 检测出的最小线段的长度
            maxLineGap=20: 最大线段间隔,小于此间隔的认为是一条直线
        @return:
            horizontal:图像中的水平线
            vertical:图像中的垂直线
        """
        # HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None) 
        #     image: 必须是二值图像,推荐使用canny边缘检测的结果图像
        #     rho: 线段以像素为单位的距离精度,double类型的,推荐用1.0 
        #     theta: 线段以弧度为单位的角度精度,推荐用numpy.pi/180 
        #     threshod: 累加平面的阈值参数,int类型,超过设定阈值才被检测出线段,值越大,基本上意味着检出的线段越长,检出的线段个数越少。根据情况推荐先用100试试
        #     lines:这个参数的意义未知,发现不同的lines对结果没影响,但是不要忽略了它的存在 
        #     minLineLength:线段以像素为单位的最小长度,根据应用场景设置 
        #     maxLineGap:同一方向上两条线段判定为一条线段的最大允许间隔（断裂）,超过了设定值,则把两条线段当成一条线段,值越大,允许线段上的断裂越大,越有可能检出潜在的直线段       
        lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 80, minLineLength=minLineLength, maxLineGap=maxLineGap)
        lines = np.squeeze(lines)
        if self.verbose:
            display_lines_on_frame(self.frame.copy(), [], lines)
        # 将直线分类为水平直线和垂直直线
        horizontal, vertical = self._classify_lines(lines)
        if self.verbose:
            display_lines_on_frame(self.frame.copy(), horizontal, vertical)
        # 将断续的直线合并
        horizontal, vertical = self._merge_lines(horizontal, vertical,minHorizontal=10, minVertical=10)
        if self.verbose:
            display_lines_on_frame(self.frame.copy(), horizontal, vertical)
        return horizontal, vertical

    def _classify_lines(self, lines, angle=26.5):
        """
        将直线分类为垂直线和水平线,通过计算直接的倾角,angle=26.5时,tanAngle=2.00057
        @parameter:
            lines: 待分类的直线
            angle: 直线倾角判断
        @return:
            clean_horizontal: 水平直线
            vertical: 垂直直线
        """
        horizontal = []
        vertical = []
        highest_vertical_y = np.inf
        lowest_vertical_y = 0
        # 计算当前倾角的tan值
        tanAngle = 1.0 / np.tan(np.deg2rad(angle)) 
        # 循环遍历，计算倾角，将直线分为水平和垂直
        for line in lines:
            x1, y1, x2, y2 = line
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx > tanAngle * dy:
                horizontal.append(line)
            else:
                vertical.append(line)
                highest_vertical_y = min(highest_vertical_y, y1, y2)
                lowest_vertical_y = max(lowest_vertical_y, y1, y2)

        # 使用垂直线条的最高点和最低点过滤垂直线
        # 将最低点距离 下移当前距离的1/15，将最高点上移但前距离的2/15，过滤不在此范围内的直线
        clean_horizontal = []
        h = lowest_vertical_y - highest_vertical_y
        lowest_vertical_y += h / 15
        # lowest_vertical_y = min(lowest_vertical_y, self.v_height/10*9)
        lowest_vertical_y = max(lowest_vertical_y, self.v_height/10*9)
        highest_vertical_y -= h * 2 / 15
        # highest_vertical_y = max(highest_vertical_y, self.v_height/10)
        highest_vertical_y = min(highest_vertical_y, self.v_height/10)

        mid_horizontal_line = (highest_vertical_y+lowest_vertical_y)/2.0
        mid_low = mid_horizontal_line - h *0.16
        mid_hight = mid_horizontal_line - h*0.42
        # print("lowest_vertical_y",lowest_vertical_y,"mid_low",mid_low,"mid_hight",mid_hight,"highest_vertical_y",highest_vertical_y)
        for line in horizontal:
            x1, y1, x2, y2 = line
            # if lowest_vertical_y > y1 > highest_vertical_y and lowest_vertical_y > y2 > highest_vertical_y:
            #     clean_horizontal.append(line)
            if (lowest_vertical_y > y1 > mid_low or mid_hight > y1 > highest_vertical_y) and (lowest_vertical_y > y2 > mid_low or mid_hight > y2 > highest_vertical_y):
                clean_horizontal.append(line)
        
        return clean_horizontal, vertical

    def _classify_vertical(self, vertical, width):
        """
        将垂直的直线分类为左边的垂直直线和右边的垂直直线
        @parmaeter
            vertical: 但前检测出的所有垂直线
            width: 图像的宽度
        @return
            vertical_lines: 过滤后的所有垂直线
            vertical_left: 左边的垂直线
            vertical_right: 右边的垂直线
        """
        vertical_lines = []
        vertical_left = []
        vertical_right = []
        right_th = width * 4 / 7    # 右边线条的起始 x 位置
        left_th = width * 3 / 7     # 左边线条的终止 x 位置

        for line in vertical:
            x1, y1, x2, y2 = line
            if x1 < left_th or x2 < left_th:
                vertical_left.append(line)
            elif x1 > right_th or x2 > right_th:
                vertical_right.append(line)
            else:
                vertical_lines.append(line)
        return vertical_lines, vertical_left, vertical_right

    def _merge_lines(self, horizontal_lines, vertical_lines, minHorizontal=10, minVertical=10):
        """
        合并线条
        @parameter:
            horizontal_lines: 所有的水平线
            vertical_lines: 所有的垂直线
            minHorizontal=10: 当两水平直线相差小于 minHorizontal 进行合并
            minVertical=10: 当两垂直直线相差小于 minVertical 进行合并
        @return
            new_horizontal_lines: 合并后的水平线
            new_vertical_lines: 合并后的垂直线
        """

        # 根据x的值将水平线进行排序
        horizontal_lines = sorted(horizontal_lines, key=lambda item: item[0])
        mask = [True] * len(horizontal_lines)
        new_horizontal_lines = []
        # 将当前直线与后面的直线一一匹配，将水平偏离像素小于10的两条线合并为一条
        for i, line in enumerate(horizontal_lines):
            if mask[i]:
                for j, s_line in enumerate(horizontal_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        dy = abs(y3 - y2)
                        if dy < minHorizontal:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[0])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False

                new_horizontal_lines.append(line)

        # 根据y值，将垂直线条进行排序
        vertical_lines = sorted(vertical_lines, key=lambda item: item[1])
        # 任取一条贯穿图像的横线
        xl, yl, xr, yr = (0, self.v_height * 6 / 7, self.v_width, self.v_height * 6 / 7)
        mask = [True] * len(vertical_lines)
        new_vertical_lines = []
        # 选择两条直线，计算两直线的于上述横线的交点
        # 判断两交点的 x 方向像素的偏移距离，距离小于10则进行合并
        
        for i, line in enumerate(vertical_lines):
            if mask[i]:
                for j, s_line in enumerate(vertical_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        xi, yi = line_intersection(((x1, y1), (x2, y2)), ((xl, yl), (xr, yr)))
                        xj, yj = line_intersection(((x3, y3), (x4, y4)), ((xl, yl), (xr, yr)))

                        dx = abs(xi - xj)
                        if dx < minVertical:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[1])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False
                new_vertical_lines.append(line)
        return new_horizontal_lines, new_vertical_lines

    def _find_homography(self, horizontal_lines, vertical_lines):
        """
        使用找到的所有直线,按排列组合,每次选取两个横线和两个竖线。求得其交点,然后使用交点与标准球场中的任意四个点进行透视计算,
        取得分最高的四个点的透视矩阵为当前图像的透视变换矩阵。
        @parameter:
            horizontal_lines:图片中检测出的水平线
            vertical_lines:图片中检测出的垂直线
        @return:
            max_mat: 标准球场到当前球场的透视变换矩阵
            max_inv_mat: 当前球场到标准球场的透视变换矩阵
            max_score: 匹配到的最好的球场的最高得分
        """
        max_score = -np.inf
        max_mat = None
        max_inv_mat = None
        # 任选两个水平线和两个垂直线，求得其交点，使用交点与标准球场中的任意四个交点计算透视矩阵
        # 使用透视矩阵将标准球场投影到当前球场，然后计算本次投影的得分
        # 选择得分最高的作为本图的投影矩阵
        count = 0
        hnum = len(list(combinations(horizontal_lines, 2)))
        vnum = len(list(combinations(vertical_lines, 2)))
        print("count:",hnum*vnum*12)
        for horizontal_pair in list(combinations(horizontal_lines, 2)):  # combinatoins 排列组合
            for vertical_pair in list(combinations(vertical_lines, 2)):
                h1, h2 = horizontal_pair
                v1, v2 = vertical_pair
                # 计算交点
                i1 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
                i2 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v2[0:2]), tuple(v2[2:])))
                i3 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
                i4 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v2[0:2]), tuple(v2[2:])))
                intersections = [i1, i2, i3, i4]
                #对交点进行排序，左上->右下
                intersections = sort_intersection_points(intersections) 
                # 循环匹配球场的任意四点
                for i, configuration in self.normal_court.court_conf.items():
                    # 计算标准球场到当前四个点的透视变化矩阵
                    matrix, _ = cv2.findHomography(np.float32(configuration), np.float32(intersections), method=0)  
                    inv_matrix = cv2.invert(matrix)[1]  # 当前矩阵到标准球场的透视变换矩阵 （matrix的逆矩阵）
                    # 计算当前四点的得分
                    confi_score = self._get_confi_score(matrix)     
                    # 选择得分最高的
                    count += 1
                    # print("h:",h1,h2,'v:',v1,v2,"i:",i,"score:",confi_score,"num:",count,'[',hnum*vnum*12,'=',hnum,'*',vnum,'*12',']')
                    if max_score < confi_score:
                        max_score = confi_score
                        max_mat = matrix
                        max_inv_mat = inv_matrix
                        self.best_conf = i
        # 可视化
        if self.verbose:
            frame = self.frame.copy()
            court = self.add_court_overlay(frame, max_mat, (255, 0, 0))
            cv2.imshow('<func:_find_homography> court', court)
            if cv2.waitKey(0) & 0xff == 27:
                cv2.destroyAllWindows()
        return max_mat, max_inv_mat, max_score

    def _get_confi_score(self, matrix):
        """
        计算透视变化的得分,将标准球场变换到当前视角,计算与当前视角检测到得网球线的重合的白色像素有多少。
        @parameter:
            matrix:透视变换矩阵
        @return
            返回 重合的像素数目 - 0.5*不重合的像素数目
        """
        court = cv2.warpPerspective(self.normal_court.court, matrix, self.frame.shape[1::-1])    # 透视变换
        court[court > 0] = 1
        gray = self.gray.copy()
        gray[gray > 0] = 1
        correct = court * gray
        wrong = court - correct
        c_p = np.sum(correct)
        w_p = np.sum(wrong)
        return c_p - 0.5 * w_p


    def add_court_overlay(self, frame, homography=None, overlay_color=(255, 255, 255), frame_num=-1):
        """
        将标准球场的网球线根据透视矩阵投影到真实图片上,并用overlay_color进行绘制
        """
        # 获取透视变换矩阵
        if homography is None and len(self.court_warp_matrix) > 0 and frame_num < len(self.court_warp_matrix):
            homography = self.court_warp_matrix[frame_num]
        # 将标准球场 透视到当前图像
        court = cv2.warpPerspective(self.normal_court.court, homography, frame.shape[1::-1])
        frame[court > 0, :] = overlay_color
        return frame

    def find_lines_location(self):
        """
        根据标准球场,将标准球场反向透视到当前frame的视角,得到每条边信息
        """
        try: 
            # 获取标准球场的所有直线参数
            self.p = np.array(self.normal_court.get_important_lines(), dtype=np.float32).reshape((-1, 1, 2))
            # 将标准球场的直线 透视 到当前图像的视角下
            self.lines = cv2.perspectiveTransform(self.p, self.court_warp_matrix[-1]).reshape(-1)
        except Exception as e:
            # print("self.normal_court.get_important_lines():",self.normal_court.get_important_lines())
            # print("e:",e)
            self.lines = []
        return self.lines
    
    def get_warped_court(self):
        """
        将标准球场图像,透视到当前frame的视角进行展示
        """
        # 将标准球场投影到当前图像
        # cv2.wrapPerspective(标准球场图片，变化矩阵，当前图像的宽高)
        court = cv2.warpPerspective(self.normal_court.court, self.court_warp_matrix[-1], self.frame.shape[1::-1])
        court[court > 0] = 1
        return court

    def track_court(self, frame, retrack_points_num=50, max_offset=6,increment=3):
        """
        对球场进行跟踪
        @parameter:
            frame: 输入图像
            retrack_points_num: 跟踪时, 检测球场的四条线,每条线采样100个点, 如果最终总点数 < retrack_points_num 则将直线偏移dist后重新计算
            max_offset: 最大偏移距离,超过此偏移距离后任然检测不到则进行重新检测。
            increment: 跟踪失败时对self.dist自增的值。
        @return
            self.new_lines: 球场的边界线
        """
        copy = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
        if self.frame_points is None:
            # 读取detection时最有匹配的点
            conf_points = np.array(self.normal_court.court_conf[self.best_conf], dtype=np.float32).reshape((-1, 1, 2))
            # 将标准球场的四个点投影到当前图片中，即获取当前图片中与标准球场最匹配的四点的坐标
            self.frame_points = cv2.perspectiveTransform(conf_points,self.court_warp_matrix[-1]).squeeze().round()
                                                        
        # 将四个点转化为四条边线
        line1 = self.frame_points[:2]
        line2 = self.frame_points[2:4]
        line3 = self.frame_points[[0, 2]]
        line4 = self.frame_points[[1, 3]]
        lines = [line1, line2, line3, line4]
        new_lines = []

        for line in lines:
            # 每条直线上采样 100 个点
            points_on_line = np.linspace(line[0], line[1], 102)[1:-1]  # 100 samples on the line
            p1 = None   # 左边界
            p2 = None   # 右边界
            p1Out = False   # 记录p1是否在图片外
            p2Out = False   # 记录p2是否在图片外
            # 判断点是否在图像外
            if line[0][0] > self.v_width or line[0][0] < 0 or line[0][1] > self.v_height or line[0][1] < 0:     
                p1Out = True
                # 找第一个再图片内的左端点
                for p in points_on_line:
                    # print("points_on_line:",points_on_line)
                    if 0 < p[0] < self.v_width and 0 < p[1] < self.v_height:   
                        p1 = p
                        break
            # 判断点是否在图像外
            if line[1][0] > self.v_width or line[1][0] < 0 or line[1][1] > self.v_height or line[1][1] < 0:
                p2Out = True
                # 序列反转，找第一个满足的右端点
                for p in reversed(points_on_line):
                    if 0 < p[0] < self.v_width and 0 < p[1] < self.v_height:
                        p2 = p
                        break
            
            # 如果直线两端点均不在图片内,直接重新检测
            if p1Out and p2Out:
                return self.detect(frame=frame)

            # 如果直线的某一个端点在图片外，则只直线保留在图片内的部分
            if p1 is not None or p2 is not None:
                print('points outside screen')
                points_on_line = np.linspace(p1 if p1 is not None else line[0], p2 if p2 is not None else line[1], 102)[
                                1:-1]

            new_points = []     # 存放找到的点
            # 寻找直线上采样点附近的点是否是白色像素
            for p in points_on_line:
                p = (int(round(p[0])), int(round(p[1])))
                top_y, top_x = max(p[1] - self.dist, 0), max(p[0] - self.dist, 0)       # 左上偏移dist距离的坐标
                bottom_y, bottom_x = min(p[1] + self.dist, self.v_height), min(p[0] + self.dist, self.v_width)       # 右下偏移dist距离的坐标
                patch = gray[top_y: bottom_y, top_x: bottom_x]      # 获取灰度图中 [top_x,top_y] 与 [bottom_x,bottom_y]坐标构成的矩形块
                y, x = np.unravel_index(np.argmax(patch), patch.shape)  # 获取patch中最大值的下标
                if patch[y, x] > 150:       # 判断最大值的颜色是否是白色
                    new_p = (x + top_x + 1, y + top_y + 1)      # 获取最大值点在图片中的坐标
                    new_points.append(new_p)
                    cv2.circle(copy, p, 1, (255, 0, 0), 1)
                    cv2.circle(copy, new_p, 1, (0, 0, 255), 1)
            new_points = np.array(new_points, dtype=np.float32).reshape((-1, 1, 2))
            # 如果四条边找到的总电数小于 new_points_num 则重新检测或者跟踪                        
            if len(new_points) < retrack_points_num:
                print('Court Shift... [operation]: ', end=' ')
                if self.dist > max_offset: # 到达最大偏移距离，重新 detect
                    print("Can't trace it. Redetect it!")
                    return self.detect(frame)
                else:   # 增大self.dist increment后 重新进行 track
                    print('Court tracking failed, adding '+str(increment)+' pixels to dist. Retrack it!')
                    self.dist += increment
                    return self.track_court(frame)
            
            # 使用检测出的点 new_points 拟合出一条新直线
            [vx, vy, x, y] = cv2.fitLine(new_points, cv2.DIST_L2, 0, 0.01, 0.01)
            new_lines.append(((int(x - vx * self.v_width), int(y - vy * self.v_width)),
                            (int(x + vx * self.v_width), int(y + vy * self.v_width))))

        # 计算四个交点
        i1 = line_intersection(new_lines[0], new_lines[2])
        i2 = line_intersection(new_lines[0], new_lines[3])
        i3 = line_intersection(new_lines[1], new_lines[2])
        i4 = line_intersection(new_lines[1], new_lines[3])
        intersections = np.array([i1, i2, i3, i4], dtype=np.float32)
        # 计算投影矩阵
        matrix, _ = cv2.findHomography(np.float32(self.normal_court.court_conf[self.best_conf]),intersections, method=0)
        inv_matrix = cv2.invert(matrix)[1]
        self.court_warp_matrix.append(matrix)
        self.game_warp_matrix.append(inv_matrix)
        self.frame_points = intersections
        # 获取标准球场直线的数据
        self.pts = np.array(self.normal_court.get_important_lines(), dtype=np.float32).reshape((-1, 1, 2))
        # 将标准球场的直线投影到当前图片视角
        self.new_lines = cv2.perspectiveTransform(self.pts, self.court_warp_matrix[-1]).reshape(-1)
        return self.new_lines

def line_intersection(line1, line2):
    """
    找到两个线的交点
    @parameter:
        line1: 第一条线,包含两个端点
        line2: 第二条线,包含两个端点
    @return:
        返回两个线的交点坐标 (交点x,交点y)
    """
    l1 = Line(line1[0], line1[1])
    l2 = Line(line2[0], line2[1])

    intersection = l1.intersection(l2)
    return intersection[0].coordinates

def sort_intersection_points(intersections):
    """
    将交点坐标按左上到右下进行排序
    @parameter:
        intersections:待排序的交点坐标
    @return:
        返回排序后的交点坐标  [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    """
    y_sorted = sorted(intersections, key=lambda x: x[1])
    p12 = y_sorted[:2]
    p34 = y_sorted[2:]
    p12 = sorted(p12, key=lambda x: x[0])
    p34 = sorted(p34, key=lambda x: x[0])
    return p12 + p34

def display_lines_on_frame(frame, horizontal=(), vertical=()):
    """
    再输入图像上绘制检测出的线条,并使用opencv进行展示.
    @parameter:
        frame: 当前图像
        horizontal: 水平线
        vertical: 垂直线
    @return
        返回绘制直线后的图像
    """

    for line in horizontal:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (x1, y1), 1, (255, 0, 0), 2)
        cv2.circle(frame, (x2, y2), 1, (255, 0, 0), 2)

    for line in vertical:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (x1, y1), 1, (255, 0, 0), 2)
        cv2.circle(frame, (x2, y2), 1, (255, 0, 0), 2)
    cv2.namedWindow('<func: display_lines_on_frame> Court', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("<func: display_lines_on_frame> Court", 640, 480)
    cv2.imshow('<func: display_lines_on_frame> Court', frame)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    return frame
