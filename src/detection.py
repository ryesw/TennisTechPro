import torch
from scipy import signal
import cv2
import numpy as np
from yolo import detection_model
from ultralytics.utils.plotting import Annotator

class DetectionModel:
    def __init__(self, match_type=2, dtype=torch.FloatTensor):
        self.detection_model = detection_model
        self.match_type = match_type # 단식 2명, 복식 4명
        self.dtype = dtype
        self.BOTTOM_PLAYER_1 = 0
        self.BOTTOM_PLAYER_2 = 2
        self.TOP_PLAYER_1 = 1
        self.TOP_PLAYER_2 = 3
        self.BOTTOM_MIN_CONF = 0.55 # Bottom 선수 Detection 시 최소 confidence 값
        self.TOP_MIN_CONF = 0.32 # Top 선수 Detection 시 최소 confidence 값
        self.line_thickness = 2
        self.player_1_boxes = []
        self.player_2_boxes = []
        self.player_1_count = 0
        self.player_2_count = 0
        self.cls = []


    def detect(self, image, person_min_score=None):
        """
        Use YOLO Model to detect players in the one frame
        """
        results = self.detection_model.predict(image)

        for result in results:
            annotator = Annotator(image, line_width=self.line_thickness, font_size=None)

            boxes = result.boxes

            # Model 성능 보고 더 간단하게 만들 것
            # 현재 모델의 Detection in Input Video1 : 2 bottom (대다수), 1 bottom 1 top (극소수), 2 bottom 1 top (극극소수)로 탐지
            # No detection은 알 바 아님
            if len(boxes.xyxy) == 3:
                box_1 = boxes.xyxy[0].cpu().numpy()
                cls_1 = boxes.cls[0].cpu().numpy()
                conf_1 = boxes.conf[0].cpu().numpy()
                cls = boxes.cls[1].cpu().numpy()
                box_2 = boxes.xyxy[2].cpu().numpy()
                cls_2 = boxes.cls[2].cpu().numpy()
                conf_2 = boxes.conf[2].cpu().numpy()
                self.player_1_boxes.append(box_1)
                self.player_2_boxes.append(box_2)
                annotator.box_label(box_1, label=False, color=(255, 0, 0))
                annotator.box_label(box_2, label=False, color=(255, 0, 0))
                self.cls.append([cls_1, cls, cls_2])

            elif len(boxes.xyxy) == 2:
                box_1 = boxes.xyxy[0].cpu().numpy()
                cls_1 = boxes.cls[0].cpu().numpy()
                conf_1 = boxes.conf[0].cpu().numpy()
                box_2 = boxes.xyxy[1].cpu().numpy()
                cls_2 = boxes.cls[1].cpu().numpy()
                conf_2 = boxes.conf[1].cpu().numpy()

                self.player_1_boxes.append(box_1)
                self.player_2_boxes.append(box_2)
                self.player_1_count += 1
                self.player_2_count += 1
                annotator.box_label(box_1, label=False, color=(255, 0, 0))
                annotator.box_label(box_2, label=False, color=(255, 0, 0))

            elif len(boxes.xyxy) == 1:
                box = boxes.xyxy[0].cpu().numpy()
                cls = boxes.cls[0].cpu().numpy()
                conf = boxes.conf[0].cpu().numpy()
                annotator.box_label(box, label=False, color=(255, 0, 0))

                if cls == 0:
                    self.player_1_boxes.append(box)
                    self.player_1_count += 1
                    self.player_2_boxes.append(None)
                elif cls == 1:
                    self.player_1_boxes.append(None)
                    self.player_2_boxes.append(box)
                    self.player_2_count += 1

            else:
                self.player_1_boxes.append(None)
                self.player_2_boxes.append(None)
                
        frame = annotator.result()
        return frame
    

    def detect_bottom_player(self, frame, court_detector):
        court_type = 1
        white_ref = court_detector.court_reference.get_court_mask(court_type)
        white_mask = cv2.warpPerspective(white_ref, court_detector.court_warp_matrix[-1], frame.shape[1::-1])

        image_court = frame.copy()
        image_court[white_mask == 0, :] = (0, 0, 0)

        results = self.detection_model.predict(image_court)

        for result in results:
            annotator = Annotator(frame, line_width=self.line_thickness, font_size=None)
            boxes = result.boxes

            if len(boxes) != 0:
                box = boxes.xyxy[0].cpu().numpy()
                cls = boxes.cls[0].cpu().numpy()
                conf = boxes.conf[0].cpu().numpy()
                self.player_1_boxes.append(box)
                annotator.box_label(box, label=False, color=(255, 0, 0))
                self.player_1_count += 1
            else:
                self.player_1_boxes.append(None)
                
        frame = annotator.result()
        return frame
    

    def detect_top_player(self, frame, court_detector):
        court_type = 2
        white_ref = court_detector.court_reference.get_court_mask(court_type)
        white_mask = cv2.warpPerspective(white_ref, court_detector.court_warp_matrix[-1], frame.shape[1::-1])
        white_mask = cv2.dilate(white_mask, np.ones((100, 1)), anchor=(0, 0))

        image_court = frame.copy()
        image_court[white_mask == 0, :] = (0, 0, 0)

        results = self.detection_model.predict(image_court)

        for result in results:
            annotator = Annotator(frame, line_width=self.line_thickness, font_size=None)
            boxes = result.boxes

            if len(boxes) != 0:
                box = boxes.xyxy[0].cpu().numpy()
                cls = boxes.cls[0].cpu().numpy()
                conf = boxes.conf[0].cpu().numpy()
                self.player_2_boxes.append(box)
                annotator.box_label(box, label=False, color=(255, 0, 0))
                self.player_2_count += 1
            else:
                self.player_2_boxes.append(None)

        frame = annotator.result()
        return frame
    
    def preprocessing_box_coordinates(self):
        positions_1 = []
        positions_2 = []

        mask_1 = []
        mask_2 = []

        for i, box in enumerate(self.player_1_boxes):
            if box is not None:
                positions_1.append(box)
                mask_1.append(True)
            elif len(positions_1) > 0:
                positions_1.append(positions_1[-1])
                mask_1.append(False)
            else:
                positions_1.append(np.array([0, 0]))
                mask_1.append(False)
            
        for i, box in enumerate(self.player_2_boxes):
            if box is not None:
                positions_2.append(box)
                mask_2.append(True)
            elif len(positions_2) > 0:
                positions_2.append(positions_2[-1])
                mask_2.append(False)
            else:
                positions_2.append(np.array([0, 0]))
                mask_2.append(False)

        # Smooth both locations
        positions_1 = np.array(positions_1)
        smoothed_1 = np.zeros_like(positions_1)
        smoothed_1[:, 0] = signal.savgol_filter(positions_1[:, 0], 7, 2)
        smoothed_1[:, 1] = signal.savgol_filter(positions_1[:, 1], 7, 2)
        smoothed_1[not mask_1, :] = [None, None]
        self.player_1_boxes = smoothed_1

        positions_2 = np.array(positions_2)
        smoothed_2 = np.zeros_like(positions_2)
        smoothed_2[:, 0] = signal.savgol_filter(positions_2[:, 0], 7, 2)
        smoothed_2[:, 1] = signal.savgol_filter(positions_2[:, 1], 7, 2)
        smoothed_2[not mask_2, :] = [None, None]
        self.player_2_boxes = smoothed_2

    def calculate_feet_positions(self, court_detector):
        """
        Calculate the feet position of both players using the inverse transformation of the court and the boxes of both players
        """
        inv_mats = court_detector.game_warp_matrix
        positions_1 = []
        positions_2 = []

        mask_1 = []
        mask_2 = []

        # Bottom player feet locations
        for i, box in enumerate(self.player_1_boxes):
            if box is not None:
                feet_pos = np.array([(box[0] + (box[2] - box[0]) / 2), box[3]]).reshape((1, 1, 2))
                feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mats[i]).reshape(-1)
                positions_1.append(feet_court_pos)
                mask_1.append(True)
            elif len(positions_1) > 0:
                positions_1.append(positions_1[-1])
                mask_1.append(False)
            else:
                positions_1.append(np.array([0, 0]))
                mask_1.append(False)

        # Top player feet locations
        for i, box in enumerate(self.player_2_boxes):
            if box is not None:
                feet_pos = np.array([(box[0] + (box[2] - box[0]) / 2), box[3]]).reshape((1, 1, 2))
                feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mats[i]).reshape(-1)
                positions_2.append(feet_court_pos)
                mask_2.append(True)
            elif len(positions_2) > 0:
                positions_2.append(positions_2[-1])
                mask_2.append(False)
            else:
                positions_2.append(np.array([0, 0]))
                mask_2.append(False)
        
        # Smooth both feet locations
        positions_1 = np.array(positions_1)
        smoothed_1 = np.zeros_like(positions_1)
        smoothed_1[:, 0] = signal.savgol_filter(positions_1[:, 0], 7, 2)
        smoothed_1[:, 1] = signal.savgol_filter(positions_1[:, 1], 7, 2)
        smoothed_1[not mask_1, :] = [None, None]

        positions_2 = np.array(positions_2)
        smoothed_2 = np.zeros_like(positions_2)
        smoothed_2[:, 0] = signal.savgol_filter(positions_2[:, 0], 7, 2)
        smoothed_2[:, 1] = signal.savgol_filter(positions_2[:, 1], 7, 2)
        smoothed_2[not mask_2, :] = [None, None] # input video 2에서는 top 선수 탐지 X -> TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'

        return smoothed_1, smoothed_2

    def get_boxes(self):
        return self.player_1_boxes, self.player_2_boxes
    
    def print_counts(self):
        print('Player 1 Object Detection Count: ', self.player_1_count)
        print('Player 2 Object Detection Count: ', self.player_2_count)


def center_of_box(box):
    """
    Calculate the center of a box
    """
    if box[0] is None:
        return None, None
    height = box[3] - box[1]
    width = box[2] - box[0]
    return box[0] + width / 2, box[1] + height / 2