import queue
import cv2
import numpy as np
from PIL import Image, ImageDraw
from TRACKNET import TrackNet


class BallDetector():
    def __init__(self):
        self.tracknet = TrackNet()
        self.tracknet.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        self.tracknet.load_weights('models/tracknet.h5')
        self.xy_coordinates = []

    def detect_ball(self, img, v_width, v_height):
        # In order to draw the trajectory of tennis, we need to save the coordinate of previous 7 frames
        q = queue.deque()
        for i in range(0, 8):
            q.appendleft(None)

        width, height = 640, 360
        n_classes = 256
        print('Detecting the ball...')

        output_img = img
        img = cv2.resize(img, (640, 360))
        img = img.astype(np.float32)

        X = np.rollaxis(img, 2, 0)
        pred = self.tracknet.predict(np.array([X]))[0]
        pred = pred.reshape((height, width, n_classes)).argmax(axis=2)
        pred = pred.astype(np.uint8)
        heatmap = cv2.resize(pred, (v_width, v_height))

        ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)
        
        PIL_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        PIL_img = Image.fromarray(PIL_img)

        if circles is not None:
            if len(circles) == 1:
                x = int(circles[0][0][0])
                y = int(circles[0][0][1])
                self.xy_coordinates.append([x, y])
                
                q.appendleft([x, y])
                q.pop()
            else:
                self.xy_coordinates.append(None)
                q.appendleft(None)
                q.pop()
        else:
            self.xy_coordinates.append(None)
            q.appendleft(None)
            q.pop()
        
        for i in range(0, 8):
            if q[i] is not None:
                x = q[i][0]
                y = q[i][1]
                bbox = (x - 2, y - 2, x + 2, y + 2)
                draw = ImageDraw.Draw(PIL_img)
                draw.ellipse(bbox, outline='yellow')
                del draw
        
        frame = cv2.cvtColor(np.array(PIL_img), cv2.COLOR_RGB2BGR)
        return frame
    
    def draw_ball_position_in_minimap(self, frame, court_detector, frame_num):
        """
        Calculate the ball position of both players using the inverse transformation of the court and the x, y positions
        """
        inv_mats = court_detector.game_warp_matrix[frame_num]
        coord = self.xy_coordinates[frame_num]
        img = frame.copy()

        # Ball locations
        if coord is not None:
            p = np.array(coord, dtype='float64')
            ball_pos = np.array([p[0], p[1]]).reshape((1, 1, 2))
            transformed = cv2.perspectiveTransform(ball_pos, inv_mats)[0][0].astype('int64')
            img = cv2.circle(img, (transformed[0], transformed[1]), 10, (0, 255, 255), -1)

        return img
    
    def calculate_xy_differences(self):
        """
        Calculate the difference between the x and y coordinates between two neighboring coordinates
        """
        diff_list = []
        for i in range(len(self.xy_coordinates) - 1):
            if self.xy_coordinates[i] is not None and self.xy_coordinates[i+1] is not None:
                pt1 = self.xy_coordinates[i]
                pt2 = self.xy_coordinates[i+1]
                diff = [abs(pt2[0] - pt1[0]), abs(pt2[1] - pt1[1])]
                diff_list.append(diff)
            else:
                diff_list.append(None)

        xx = np.array([diff[0] if diff is not None else np.nan for diff in diff_list])
        yy = np.array([diff[1] if diff is not None else np.nan for diff in diff_list])
        return xx, yy

    def remove_outliers(self):
        xx, yy = self.calculate_xy_differences()
        indices = set(np.where(xx > 50)[0]) & set(np.where(yy > 50)[0])
        for idx in indices:
            left, middle, right = self.xy_coordinates[idx - 1], self.xy_coordinates[idx], self.xy_coordinates[idx + 1]
            if left is None:
                left = [0]
            if right is None:
                right = [0]
            if middle is None:
                middle = [0]
            max_value = max(map(list, (left, middle, right)))
            if max_value == [0]:
                pass
            else:
                try:
                    self.xy_coordinates[self.xy_coordinates.index(tuple(max_value))] = None
                except ValueError:
                    self.xy_coordinates[self.xy_coordinates.index(max_value)] = None
                    
    def interpolate_coords(self):
        xx = np.array([coord[0] if coord is not None else np.nan for coord in self.xy_coordinates])
        yy = np.array([coord[1] if coord is not None else np.nan for coord in self.xy_coordinates])

        nans, y = np.isnan(xx), lambda z: z.nonzero()[0]
        xx[nans] = np.interp(y(nans), y(~nans), xx[~nans])

        nans, x = np.isnan(yy), lambda z: z.nonzero()[0]
        yy[nans] = np.interp(x(nans), x(~nans), yy[~nans])

        self.xy_coordinates = [*zip(xx, yy)]