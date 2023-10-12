import os
import cv2
import torch
import numpy as np
import pandas as pd

from yolo import pose_model
from ultralytics.utils.plotting import Annotator


class PoseExtractor:
    def __init__(self, person_num=2, box=False, dtype=torch.FloatTensor):
        self.pose_model = pose_model # YOLO Pose 모델 추가
        self.dtype = dtype
        self.person_num = person_num  # 단식: 2명, 복식: 4명
        self.box = box # box 표시할건지
        self.PERSON_LABEL = None
        self.SCORE_MIN = 0.9
        self.keypoint_threshold = 2
        self.data = []
        self.margin = 50
        self.line_connection = [(7, 9), (7, 5), (10, 8), (8, 6), (6, 5), (15, 13),
                                (13, 11), (11, 12), (12, 14), (14, 16), (5, 11), (12, 6)] # Court Line
        self.COCO_PERSON_KEYPOINT_NAMES = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
            'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle'
        ]
        self.AIHUB_PERSON_KEYPOINT_NAMES = [
            'head', 'neck', 'chst', 'bely', 'sdlf', 'sdrt', 'eblf', 'ebrt', 'wtlf', 'wtrt', 'hplf', 'hprt', 'knlf', 'knrt', 'aklf', 'akrt'
            ]
        self.player = ['target']


    def extract_pose(self, image, p1_boxes, p2_boxes):
        """
        Extract poses from the given image for both players
        """
        p1_img = self._extract_player1_pose(image, p1_boxes)
        result = self._extract_player2_pose(p1_img, p2_boxes)

        return result


    def _extract_player1_pose(self, image, p1_boxes):
        """
        Extract the pose of player 1(bottom) from the given image
        """
        height, width = image.shape[:2]
        
        if len(p1_boxes) > 0:
            xt, yt, w, h = p1_boxes[-1]
            xt, yt, xb, yb = int(xt), int(yt), int(xt + w), int(yt + h)
            patch = image[max(yt - self.margin, 0):min(yb + self.margin, height), max(xt - self.margin, 0):min(xb + self.margin, width)].copy() # copy 안하면 오류남^^

            p1_patch = self._annotate_pose_on_patch(patch)

            image[max(yt - self.margin, 0):min(yb + self.margin, height), max(xt - self.margin, 0):min(xb + self.margin, width)] = p1_patch

        return image


    def _extract_player2_pose(self, image, p2_boxes):
        """
        Extract the pose of player 2(top) from the given image
        """
        height, width = image.shape[:2]

        if len(p2_boxes) > 0:
            xt, yt, w, h = p2_boxes[-1]
            xt, yt, xb, yb = int(xt), int(yt), int(xt + w), int(yt + h)
            patch = image[max(yt - self.margin, 0):min(yb + self.margin, height), max(xt - self.margin, 0):min(xb + self.margin, width)].copy() # copy 안하면 오류남^^

            p2_patch = self._annotate_pose_on_patch(patch)
            
            image[max(yt - self.margin, 0):min(yb + self.margin, height), max(xt - self.margin, 0):min(xb + self.margin, width)] = p2_patch

        return image


    def _annotate_pose_on_patch(self, patch):
        """
        Annotate the pose on a patch of the image
        """
        results = self.pose_model.predict(patch, boxes=False)
        for result in results:
            annotator = Annotator(patch, line_width=3, pil=True)
            kpts = result.keypoints # get box coordinates in (top, left, bottom, right) format

            for kpt in kpts:
                pt = kpt.data[0].cpu().numpy()
                annotator.kpts(pt)

        player_patch = annotator.result()
        return player_patch


    def save_to_csv(self, output_folder):
        """
        Saves the pose keypoints data as csv
        :param output_folder: str, path to output folder
        :return: df, the data frame of the pose keypoints
        """
        columns = self.AIHUB_PERSON_KEYPOINT_NAMES # self.COCO_PERSON_KEYPOINT_NAMES
        columns_x = [column + '_x' for column in columns]
        columns_y = [column + '_y' for column in columns]
        df = pd.DataFrame(self.data, columns=columns_x + columns_y) # + self.player
        outfile_path = os.path.join(output_folder, 'stickman_data.csv')
        df.to_csv(outfile_path, index=False)
        return df