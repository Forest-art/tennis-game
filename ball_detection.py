import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy import signal
import os
from scipy.signal import find_peaks
import math 
from ball_tracker_net import BallTrackerNet
from detection import center_of_box
from utils import get_video_properties
from court_reference import CourtReference


def combine_three_frames(frame1, frame2, frame3, width, height):
    """
    Combine three frames into one input tensor for detecting the ball
    """

    # Resize and type converting for each frame
    img = cv2.resize(frame1, (width, height))
    # input must be float type
    img = img.astype(np.float32)

    # resize it
    img1 = cv2.resize(frame2, (width, height))
    # input must be float type
    img1 = img1.astype(np.float32)

    # resize it
    img2 = cv2.resize(frame3, (width, height))
    # input must be float type
    img2 = img2.astype(np.float32)

    # combine three imgs to  (width , height, rgb*3)
    imgs = np.concatenate((img, img1, img2), axis=2)

    # since the odering of TrackNet  is 'channels_first', so we need to change the axis
    imgs = np.rollaxis(imgs, 2, 0)
    return np.array(imgs)


class BallDetector:
    """
    Ball Detector model responsible for receiving the frames and detecting the ball
    """
    def __init__(self, save_state, out_channels=2):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Load TrackNet model weights
        self.detector = BallTrackerNet(out_channels=out_channels)
        saved_state_dict = torch.load(save_state)
        self.detector.load_state_dict(saved_state_dict['model_state'])
        self.detector.eval().to(self.device)

        self.current_frame = None
        self.last_frame = None
        self.before_last_frame = None

        self.video_width = None
        self.video_height = None
        self.model_input_width = 640
        self.model_input_height = 360

        self.threshold_dist = 100
        self.xy_coordinates = np.array([[None, None], [None, None]])

        self.bounces_indices = []

    def detect_ball(self, frame):
        """
        After receiving 3 consecutive frames, the ball will be detected using TrackNet model
        :param frame: current frame
        """
        # Save frame dimensions
        if self.video_width is None:
            self.video_width = frame.shape[1]
            self.video_height = frame.shape[0]
        self.last_frame = self.before_last_frame
        self.before_last_frame = self.current_frame
        self.current_frame = frame.copy()

        # detect only in 3 frames were given
        if self.last_frame is not None:
            # combine the frames into 1 input tensor
            frames = combine_three_frames(self.current_frame, self.before_last_frame, self.last_frame,
                                          self.model_input_width, self.model_input_height)
            frames = (torch.from_numpy(frames) / 255).to(self.device)
            # Inference (forward pass)
            x, y = self.detector.inference(frames)
            if x is not None:
                # Rescale the indices to fit frame dimensions
                x = x * (self.video_width / self.model_input_width)
                y = y * (self.video_height / self.model_input_height)

                # Check distance from previous location and remove outliers
                if self.xy_coordinates[-1][0] is not None:
                    if np.linalg.norm(np.array([x,y]) - self.xy_coordinates[-1]) > self.threshold_dist:
                        x, y = None, None
            self.xy_coordinates = np.append(self.xy_coordinates, np.array([[x, y]]), axis=0)

    def mark_positions(self, frame, mark_num=4, frame_num=None, ball_color='yellow'):
        """
        Mark the last 'mark_num' positions of the ball in the frame
        :param frame: the frame we mark the positions in
        :param mark_num: number of previous detection to mark
        :param frame_num: current frame number
        :param ball_color: color of the marks
        :return: the frame with the ball annotations
        """
        bounce_i = None
        # if frame number is not given, use the last positions found
        if frame_num is not None:
            q = self.xy_coordinates[frame_num-mark_num+1:frame_num+1, :]
            for i in range(frame_num - mark_num + 1, frame_num + 1):
                if i in self.bounces_indices:
                    bounce_i = i - frame_num + mark_num - 1
                    break
        else:
            q = self.xy_coordinates[-mark_num:, :]
        pil_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(pil_image)
        # Mark each position by a circle
        for i in range(q.shape[0]):
            if q[i, 0] is not None:
                draw_x = q[i, 0]
                draw_y = q[i, 1]
                bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
                draw = ImageDraw.Draw(pil_image)
                if bounce_i is not None and i == bounce_i:
                    draw.ellipse(bbox, outline='red')
                else:
                    draw.ellipse(bbox, outline=ball_color)

            # Convert PIL image format back to opencv image format
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return frame

    def show_y_graph(self, player_1_boxes, player_2_boxes):
        """
        Display ball y index positions and both players y index positions in all the frames in a graph
        :param player_1_boxes: bottom player boxes
        :param player_2_boxes: top player boxes
        """
        player_1_centers = np.array([center_of_box(box) for box in player_1_boxes])
        player_1_y_values = player_1_centers[:, 1]
        # get y value of top quarter of bottom player box
        player_1_y_values -= np.array([(box[3] - box[1]) // 4 for box in player_1_boxes])

        # Calculate top player boxes center
        player_2_centers = []
        for box in player_2_boxes:
            if box[0] is not None:
                player_2_centers.append(center_of_box(box))
            else:
                player_2_centers.append([None, None])
        player_2_centers = np.array(player_2_centers)
        player_2_y_values = player_2_centers[:, 1]

        y_values = self.xy_coordinates[:, 1].copy()
        x_values = self.xy_coordinates[:, 0].copy()

        plt.figure()
        plt.scatter(range(len(y_values)), y_values)
        plt.plot(range(len(player_1_y_values)), player_1_y_values, color='r')
        plt.plot(range(len(player_2_y_values)), player_2_y_values, color='g')
        plt.show()



def is_in_poly(p, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
    px, py = p
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner
        x2, y2 = poly[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in


# 坐标转换
def cvt_pos(pos, cvt_mat_t):
    u = pos[0]
    v = pos[1]
    x = (cvt_mat_t[0][0]*u+cvt_mat_t[0][1]*v+cvt_mat_t[0][2])/(cvt_mat_t[2][0]*u+cvt_mat_t[2][1]*v+cvt_mat_t[2][2])
    y = (cvt_mat_t[1][0]*u+cvt_mat_t[1][1]*v+cvt_mat_t[1][2])/(cvt_mat_t[2][0]*u+cvt_mat_t[2][1]*v+cvt_mat_t[2][2])

    return [int(x), int(y)]


# 网球场主场尺寸10.97 * 23.77m
def dist(p1, p2):
    dx = abs(p1[0] - p2[0]) / 360 * 10.97
    dy = abs(p1[1] - p2[1]) / 700 * 23.77
    return math.sqrt(dx * dx + dy * dy)



# Caluate the speed of tennis, detect whether the ball is out court or hit the ground
def speed(points, fps, court_points):
    x = points[:, 0]
    y = points[:, 1]
    x = np.array([0 if i is None else i for i in x])
    y = np.array([0 if i is None else i for i in y])

    hit_ground = 0
    out_court = 0
    
    M = cv2.getPerspectiveTransform(court_points, np.float32([[0, 0], [360, 0], [360, 700], [0, 700]]))

    hit_point = [0, 0]
    for i in range(len(y)):
        if i > 1 and i < len(y) - 2 and y[i-2] < y[i-1] and y[i-1] < y[i] and y[i] > y[i+1] and y[i+1] > y[i+2]:
            if is_in_poly([x[i], y[i]], court_points):
                hit_ground = 1
                hit_point = cvt_pos([x[i], y[i]], M)
            if x[i] != 0 and y[i] != 0:
                if not is_in_poly([x[i], y[i]], court_points):
                    out_court = 1

    if len(x[x>0]) < 2:
        return -1, hit_ground, out_court, hit_point
    
    speed_list = []
    detected_idx = np.where(x)[0]
    for i in range(len(detected_idx)):
        if i > 0:
            # if is_in_poly([x[detected_idx[i]], y[detected_idx[i]]], court_points) and is_in_poly([x[detected_idx[i-1]], y[detected_idx[i-1]]], court_points):
                # p1 = cvt_pos([x[detected_idx[i]], y[detected_idx[i]]], M)
                # p2 = cvt_pos([x[detected_idx[i-1]], y[detected_idx[i-1]]], M)
            p1 = [x[detected_idx[i]], y[detected_idx[i]]]
            p2 = [x[detected_idx[i-1]], y[detected_idx[i-1]]]
            # print([p1, p2], [[x[detected_idx[i]], y[detected_idx[i]]], [x[detected_idx[i-1]], y[detected_idx[i-1]]]])
            # print(dist(p1, p2), dist([x[detected_idx[i]], y[detected_idx[i]]], [x[detected_idx[i-1]], y[detected_idx[i-1]]]))
            speed_cur = dist(p1, p2) * fps / (detected_idx[i] - detected_idx[i-1])
            speed_list.append(speed_cur)
    
    if len(speed_list) == 0:
        return -1, hit_ground, out_court, hit_point

    return sum(speed_list) / len(speed_list), hit_ground, out_court, hit_point



def create_top_view():
    """
    Creates top view video of the gameplay
    """
    court_reference = CourtReference()
    court = court_reference.court.copy()
    court = cv2.line(court, *court_reference.net, 255, 5)
    v_width, v_height = court.shape[::-1]
    court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)
    
    return court


def ball_detect(input_video, output, weights):
    ball_detector = BallDetector(weights)
    cap = cv2.VideoCapture(input_video)
    # get videos properties
    fps, length, v_width, v_height = get_video_properties(cap)

    os.makedirs(output, exist_ok=True)
    out = cv2.VideoWriter(os.path.join(output, input_video.split('/')[-1].replace('mp4','avi')),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (v_width, v_height))

    dic = np.load("test3.npy",allow_pickle=True)
    court_points = np.array([dic.item()["topBaseLine"][0], dic.item()["topBaseLine"][1], dic.item()["bottomBaseLine"][1], dic.item()["bottomBaseLine"][0]])
    print(court_points)


    frame_i = 0
    while True:

        court = create_top_view()
        court = court[561:2935, 286:1379]
        court_resized = cv2.resize(court, (700, 360), interpolation = cv2.INTER_AREA)

        ret, frame = cap.read()
        frame_i += 1
        if not ret:
            break

        ball_detector.detect_ball(frame)

        if len(ball_detector.xy_coordinates) >= 10:
            ball_clip = ball_detector.xy_coordinates[-10:]
        else:
            ball_clip = ball_detector.xy_coordinates
        
        speeds, hit, out_court, hit_point = speed(ball_clip, fps, court_points)
        print(ball_clip, speeds, hit, out_court, hit_point)
        if hit == 1:
            court_resized = cv2.circle(court_resized, (int(hit_point[1]), int(hit_point[0])), 6, (0, 0, 255), -1)
        
        x = ball_detector.xy_coordinates[-1][0] if ball_detector.xy_coordinates[-1][0] is not None else 0
        y = ball_detector.xy_coordinates[-1][1] if ball_detector.xy_coordinates[-1][0] is not None else 0



        if speeds != -1:
            cv2.putText(frame, 'V: {:.2f}m/s'.format(speeds),
                    (max(0, int(x) - 10), max(0, int(y) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if hit == 1:
            cv2.putText(frame, 'Hit-Ground', (min(v_width, int(x) + 10), min(v_height, int(y) + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        if out_court == 1:
            cv2.putText(frame, 'Out-Courtline', (min(v_width, int(x) + 10), min(v_height, int(y) + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        frame[0:700, v_width-360:v_width] = court_resized.transpose(1,0,2)


        img = ball_detector.mark_positions(frame, frame_num=frame_i)
        out.write(img)

        cv2.imwrite("demo/test.jpg", img)


    cap.release()
    out.release()
    cv2.destroyAllWindows()




    










if __name__ == "__main__":

    


    ball_detect('videos/test3.mp4', 'demo', 'saved states/tracknet_weights_2_classes.pth')


    # from scipy.interpolate import interp1d

    # y_values = ball_detector.xy_coordinates[:,1]

    # new = signal.savgol_filter(y_values, 3, 2)

    # x = np.arange(0, len(new))
    # indices = [i for i, val in enumerate(new) if np.isnan(val)]
    # x = np.delete(x, indices)
    # y = np.delete(new, indices)
    # f = interp1d(x, y, fill_value="extrapolate")
    # f2 = interp1d(x, y, kind='cubic', fill_value="extrapolate")
    # xnew = np.linspace(0, len(y_values), num=len(y_values), endpoint=True)
    # plt.plot(np.arange(0, len(new)), new, 'o',xnew,
    #          f2(xnew), '-r')
    # plt.legend(['data', 'inter'], loc='best')
    # plt.show()

    # positions = f2(xnew)
    # peaks, _ = find_peaks(positions, distance=30)
    # a = np.diff(peaks)
    # plt.plot(positions)
    # plt.plot(peaks, positions[peaks], "x")
    # plt.show()