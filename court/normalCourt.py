import cv2
import numpy as np
import matplotlib.pyplot as plt
import os




class NormalCourt:

    """
    标准球场类
    包括标准球场的详细参数坐标,绘制球场以及球场四点参考图,球场尺寸标注图
    """
    def __init__(self):

        self.path = os.path.dirname(__file__)   # 获取存放标准球场图片的位置，与当前py文件同一个目录

        # # 网球场基本参数
        self.startPoint = (500,100)   # 起始绘制点,做上角点
        self.courtWidth = 1097  # 短边长（水平方向的边、底边）
        self.courtHeight = 2377  # 长边长（竖直方向的边，侧边）
        l_i_w = 137 # 单打线和双打线的长（竖直方向的第一条边和竖直方向的第二条边）
        l_i_h = 549 # 底边到内横线的距离 548.5
        

        self.topBaseLine = ((self.startPoint[0], self.startPoint[1]), (self.startPoint[0]+self.courtWidth, self.startPoint[1]))    # 上边界
        self.bottomBaseLine = ((self.startPoint[0], self.startPoint[1]+self.courtHeight), (self.startPoint[0]+self.courtWidth, self.startPoint[1]+self.courtHeight))     # 下横边界
        self.net = ((self.startPoint[0], int(self.startPoint[1]+self.courtHeight/2.0)), (self.startPoint[0]+self.courtWidth, int(self.startPoint[1]+self.courtHeight/2.0)))   # 网球场中网
        self.leftOuterLine =((self.startPoint[0], self.startPoint[1]), (self.startPoint[0], self.startPoint[1]+self.courtHeight))      # 左侧外边
        self.rightOuterLine = ((self.startPoint[0]+self.courtWidth, self.startPoint[1]), (self.startPoint[0]+self.courtWidth, self.startPoint[1]+self.courtHeight))    # 右侧外边
        self.leftInnerLine = ((self.startPoint[0]+l_i_w, self.startPoint[1]), (self.startPoint[0]+l_i_w, self.startPoint[1]+self.courtHeight))     #左侧内边
        self.rightInnerLine = ((self.startPoint[0]+self.courtWidth-l_i_w, self.startPoint[1]), (self.startPoint[0]+self.courtWidth-l_i_w, self.startPoint[1]+self.courtHeight))    # 右侧内边
        self.middleLine = ((int(self.startPoint[0]+self.courtWidth/2.0), self.startPoint[1]+l_i_h), (int(self.startPoint[0]+self.courtWidth/2.0), self.startPoint[1]+self.courtHeight-l_i_h))   # 中线
        self.topInnerLine = ((self.startPoint[0]+l_i_w, self.startPoint[1]+l_i_h), (self.startPoint[0]+self.courtWidth-l_i_w, self.startPoint[1]+l_i_h))      # 上横内边界
        self.bottomInnerLine = ((self.startPoint[0]+l_i_w, self.startPoint[1]+self.courtHeight-l_i_h), (self.startPoint[0]+self.courtWidth-l_i_w, self.startPoint[1]+self.courtHeight-l_i_h))   # 下横内边界

        # 任选球场线交点中的四个，用于图像匹配
        # self.court_conf = {1: [*self.topBaseLine, *self.bottomBaseLine],
        #                    2: [self.leftInnerLine[0], self.rightInnerLine[0], self.leftInnerLine[1],
        #                        self.rightInnerLine[1]],
        #                    3: [self.leftInnerLine[0], self.rightOuterLine[0], self.leftInnerLine[1],
        #                        self.rightOuterLine[1]],
        #                    4: [self.leftOuterLine[0], self.rightInnerLine[0], self.leftOuterLine[1],
        #                        self.rightInnerLine[1]],
        #                    5: [*self.topInnerLine, *self.bottomInnerLine],
        #                    6: [*self.topInnerLine, self.leftInnerLine[1], self.rightInnerLine[1]],
        #                    7: [self.leftInnerLine[0], self.rightInnerLine[0], *self.bottomInnerLine],
        #                    8: [self.rightInnerLine[0], self.rightOuterLine[0], self.rightInnerLine[1],
        #                        self.rightOuterLine[1]],
        #                    9: [self.leftOuterLine[0], self.leftInnerLine[0], self.leftOuterLine[1],
        #                        self.leftInnerLine[1]],
        #                    10: [self.topInnerLine[0], self.middleLine[0], self.bottomInnerLine[0],
        #                         self.middleLine[1]],
        #                    11: [self.middleLine[0], self.topInnerLine[1], self.middleLine[1],
        #                         self.bottomInnerLine[1]],
        #                    12: [*self.bottomInnerLine, self.leftInnerLine[1], self.rightInnerLine[1]]}

        # self.court_conf = {1: [(self.topBaseLine[0][0],self.bottomInnerLine[0][1]),self.bottomBaseLine[0],
        #                         (self.topBaseLine[1][0],self.bottomInnerLine[1][1]),self.bottomBaseLine[1]],
        #                     2: [*self.bottomInnerLine,
        #                     (self.bottomInnerLine[0][0],self.bottomBaseLine[0][1]),(self.bottomInnerLine[1][0],self.bottomBaseLine[1][1])],
        #                     3: [self.bottomInnerLine[0],(self.bottomBaseLine[1][0],self.bottomInnerLine[1][1]),
        #                        (self.bottomInnerLine[0][0],self.bottomBaseLine[0][1]),self.bottomBaseLine[1]],
        #                     4: [(self.bottomBaseLine[0][0],self.bottomInnerLine[0][1]),self.bottomInnerLine[1],
        #                          self.bottomBaseLine[0],(self.bottomInnerLine[1][0],self.bottomBaseLine[1][1])],
        #                     5: [self.bottomInnerLine[1],(self.bottomBaseLine[1][0],self.bottomInnerLine[0][1]),
        #                        (self.bottomInnerLine[1][0],self.bottomBaseLine[1][1]), self.bottomBaseLine[1]],
        #                     6: [(self.bottomBaseLine[0][0],self.bottomInnerLine[0][1]),self.bottomInnerLine[0],
        #                          self.bottomBaseLine[0], (self.bottomInnerLine[0][0],self.bottomBaseLine[0][1])]
        #                    }
        self.court_conf = {1:  [*self.bottomInnerLine,
                            (self.bottomInnerLine[0][0],self.bottomBaseLine[0][1]),(self.bottomInnerLine[1][0],self.bottomBaseLine[1][1])],
                           }
        self.line_width = 3     # 球场线宽度

        self.courtImageWidth = self.courtWidth + self.startPoint[0] * 2      # 球场图片的宽度
        self.courtImageHeight = self.courtHeight + self.startPoint[1] * 2    # 球场图片的高度

        # 判断标准球场配置文件是否存在，否则进行重新生成
        if not self.isExist():
            os.makedirs(os.path.join(self.path,"court_configurations"), exist_ok=True)
            self.build_court_reference()    # 创建黑白球场参考
            self.court = cv2.cvtColor(cv2.imread(os.path.join(self.path,'court_configurations/court_reference.png')), cv2.COLOR_BGR2GRAY)   # 读取球场信息
            self.save_all_court_configurations()    # 创建球场四点匹配参考图
        else:
            self.court = cv2.cvtColor(cv2.imread(os.path.join(self.path,'court_configurations/court_reference.png')), cv2.COLOR_BGR2GRAY)    # 读取球场信息

    def isExist(self):
        """
        判断球场参考文件夹【court_configurations】是否存在,以及内部文件是否完整
        """
        if os.path.exists(os.path.join(self.path,"court_configurations")):
            if not os.path.exists(os.path.join(self.path,'court_configurations','court_reference.png')):
                    return False
            for i in self.court_conf.keys():
                if not os.path.exists(os.path.join(self.path, 'court_configurations','court_conf_'+str(i)+'.png')) :
                    return False
        else:
            return False
        return True

    def build_court_reference(self):
        """
        根据实际球场,创建标准球场
        """
        court = np.zeros((self.courtImageHeight, self.courtImageWidth), dtype=np.uint8)
        cv2.line(court, *self.topBaseLine, 1, self.line_width)
        cv2.line(court, *self.bottomBaseLine, 1, self.line_width)
        # cv2.line(court, *self.net, 1, self.line_width)
        cv2.line(court, *self.topInnerLine, 1, self.line_width)
        cv2.line(court, *self.bottomInnerLine, 1, self.line_width)
        cv2.line(court, *self.leftOuterLine, 1, self.line_width)
        cv2.line(court, *self.rightOuterLine, 1, self.line_width)
        cv2.line(court, *self.leftInnerLine, 1, self.line_width)
        cv2.line(court, *self.rightInnerLine, 1, self.line_width)
        cv2.line(court, *self.middleLine, 1, self.line_width)
        court = cv2.dilate(court, np.ones((5, 5), dtype=np.uint8))
        plt.imsave(os.path.join(self.path,'court_configurations/court_reference.png'), court, cmap='gray')
        self.court = court
        return court

    def get_important_lines(self):
        """
        返回球场的所有边界线
        """
        lines = [*self.topBaseLine, *self.bottomBaseLine, *self.net, *self.leftOuterLine, *self.rightOuterLine,
                 *self.leftInnerLine, *self.rightInnerLine, *self.middleLine,
                 *self.topInnerLine, *self.bottomInnerLine]
        return lines

    def save_all_court_configurations(self):
        """
        绘制并保存球场的四点参考图
        """
        for i, conf in self.court_conf.items():
            c = cv2.cvtColor(255 - self.court, cv2.COLOR_GRAY2BGR)
            for p in conf:
                c = cv2.circle(c, p, 15, (0, 0, 255), 30)
            cv2.imwrite(os.path.join(self.path,f'court_configurations/court_conf_{i}.png'), c)

    def get_court_mask(self, mask_type=0):
        """
        获取球场的掩码
        """
        mask = np.ones_like(self.court)
        if mask_type == 1:  # Bottom half court，下半场
            mask[self.topBaseLine[0][1]:self.net[0][1], self.topBaseLine[0][0]:self.topBaseLine[1][0]] = 0
        elif mask_type == 2:  # Top half court，上半场
            mask[self.net[0][1]:self.bottomBaseLine[0][1],self.bottomBaseLine[0][0]:self.bottomBaseLine[1][0]] = 0
        elif mask_type == 3: # court without margins，无边界球场
            mask[:self.topBaseLine[0][1], :] = 0
            mask[self.bottomBaseLine[0][1]:, :] = 0
            mask[:, :self.leftOuterLine[0][0]] = 0
            mask[:, self.rightOuterLine[0][0]:] = 0
        # plt.imsave("mask.jpg",mask)
        return mask

    def draw_court_withLength(self):
        """
        绘制标准球场的代参数的图片
        """
        ey = 30
        def drawline(img,linetuple,name=""):
            """
            绘制直线并进行文本的标注
            """
            nonlocal ey
            linetuple = ((int(linetuple[0][0]),int(linetuple[0][1])),(int(linetuple[1][0]),int(linetuple[1][1])))
            cv2.line(img, *linetuple, 1, self.line_width)
            if abs(linetuple[0][0] - linetuple[1][0])!=0:  # 水平
                x = int((linetuple[0][0] + linetuple[1][0])/2.0)
                y = linetuple[0][1]
                l = abs(linetuple[0][0] - linetuple[1][0])
                cv2.putText(img,name+":"+str(l),(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=1, thickness=1)
            else:
                y = int((linetuple[0][1] + linetuple[1][1])/2.0)+ey
                x = linetuple[0][0]
                l = abs(linetuple[0][1] - linetuple[1][1])
                cv2.putText(img, name+":"+str(l), (x+5,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=1, thickness=1)
                ey += 30
            # print(linetuple, "{} x:{} y:{} length:{}".format(name,x,y,l))
        court = np.zeros((self.courtImageHeight, self.courtImageWidth), dtype=np.uint8)
        drawline(court,self.topBaseLine,name="topBaseLine")
        drawline(court,self.bottomBaseLine,name="bottomBaseLine")
        drawline(court,self.net,name="net")
        drawline(court,self.leftOuterLine,name="leftOuterLine")
        drawline(court,self.rightOuterLine,name="rightOuterLine")
        drawline(court,self.leftInnerLine,name="leftInnerLine")
        drawline(court,self.rightInnerLine,name="rightInnerLine")
        drawline(court,self.middleLine,name="middleLine")
        drawline(court,self.topInnerLine,name="topInnerLine")
        drawline(court,self.bottomInnerLine,name="bottomInnerLine")
        court = cv2.dilate(court, np.ones((5, 5), dtype=np.uint8))
        plt.imsave(os.path.join(self.path,"court_configurations/court_withLength.png"),court, cmap='gray')

if __name__ == '__main__':
    c = NormalCourt()
    print(c.court_conf)
    # c.get_court_mask(3)
    c.draw_court_withLength()
    # print(c.isExist())
    # c.build_court_reference()



