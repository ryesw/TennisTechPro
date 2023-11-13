import numpy as np
from scipy.signal import find_peaks

from utils import center_of_box

from tensorflow.keras.models import load_model


class ActionRecognition:
    def __init__(self):
        self.model = load_model('models/best.h5')
        self.seq_length = 60
        self.motions = ['backhand', 'forehand', 'service', 'smash']

    def predict(self, p1_boxes, p2_boxes, p1_keypoints_df, p2_keypoints_df, ball_positions, total_frame):
        p1_motion_frames = find_p1_motion_frame(p1_boxes, ball_positions, p1_keypoints_df)
        p2_motion_frames = find_p2_motion_frame(p2_boxes, ball_positions, p2_keypoints_df)
        print(p1_motion_frames)
        print(p2_motion_frames)
        # y_pred = self.model.predict()
        for frame_num in p1_motion_frames:
            p1_kpts = self.get_p1_keypoints(frame_num, p1_keypoints_df, total_frame)


        for frame_num in p2_motion_frames:
            p2_kpts = self.get_p1_keypoints(frame_num, p2_keypoints_df, total_frame)


    def get_p1_keypoints(self, frame_num, p1_keypoints_df, total_frame):
        p1_keypoints = p1_keypoints_df.values

        before_row = max(0, frame_num - int(self.seq_length / 2 + 1))
        after_row = max(frame_num +int(self.seq_length / 2), total_frame)
        kpts = p1_keypoints[before_row : after_row]
        
        return kpts




def find_p1_motion_frame(p1_boxes, ball_positions, p1_keypoints_df):
    left_wrist_index = 18
    right_wrist_index = 20
    left_wrist_pos = p1_keypoints_df.iloc[:, [left_wrist_index, left_wrist_index+1]].values
    right_wrist_pos = p1_keypoints_df.iloc[:, [right_wrist_index, right_wrist_index+1]].values

    peaks, _ = find_peaks(ball_positions[:, 1])

    dists = []
    for i, p1_box in enumerate(p1_boxes):
        if p1_box is not None:
            player_center = center_of_box(p1_box)
            ball_pos = np.array(ball_positions[i, :])
            box_dist = np.linalg.norm(player_center - ball_pos)
            left_wrist_dist, right_wrist_dist = np.inf, np.inf

            if left_wrist_pos[i, 0] > 0:
                left_wrist_dist = np.linalg.norm(left_wrist_pos[i, :] - ball_pos)

            if right_wrist_pos[i, 0] > 0:
                right_wrist_dist = np.linalg.norm(right_wrist_pos[i, :] - ball_pos)

            dists.append(min(box_dist, left_wrist_dist, right_wrist_dist))
        else:
            dists.append(None)
    dists = np.array(dists)

    p1_motion_indices = []
    print(peaks)
    for peak in peaks:
        player_box_height = max(p1_boxes[peak][3] - p1_boxes[peak][1], 130)
        if dists[peak] < (player_box_height * 4 / 5):
            p1_motion_indices.append(peak)

    while True:
        diffs = np.diff(p1_motion_indices)
        to_del = []
        for i, diff in enumerate(diffs):
            if diff < 40:
                max_in = np.argmax([dists[p1_motion_indices[i]], dists[p1_motion_indices[i + 1]]])
                to_del.append(i + max_in)

        p1_motion_indices = np.delete(p1_motion_indices, to_del)
        if len(to_del) == 0:
            break

    return p1_motion_indices

def find_p2_motion_frame(p2_boxes, ball_positions, p2_keypoints_df):
    left_wrist_index = 18
    right_wrist_index = 20
    left_wrist_pos = p2_keypoints_df.iloc[:, [left_wrist_index, left_wrist_index+1]].values
    right_wrist_pos = p2_keypoints_df.iloc[:, [right_wrist_index, right_wrist_index+1]].values

    peaks, _ = find_peaks(ball_positions[:, 1] * (-1))
    print(peaks)
    dists = []
    for i, p1_box in enumerate(p2_boxes):
        if p1_box is not None:
            player_center = center_of_box(p1_box)
            ball_pos = np.array(ball_positions[i, :])
            box_dist = np.linalg.norm(player_center - ball_pos)
            left_wrist_dist, right_wrist_dist = np.inf, np.inf

            if left_wrist_pos[i, 0] > 0:
                left_wrist_dist = np.linalg.norm(left_wrist_pos[i, :] - ball_pos)

            if right_wrist_pos[i, 0] > 0:
                right_wrist_dist = np.linalg.norm(right_wrist_pos[i, :] - ball_pos)

            dists.append(min(box_dist, left_wrist_dist, right_wrist_dist))
        else:
            dists.append(None)
    dists = np.array(dists)

    p2_motion_indices = []
    for peak in peaks:
        player_box_height = max(p2_boxes[peak][3] - p2_boxes[peak][1], 130)
        if dists[peak] < (player_box_height * 4 / 5):
            p2_motion_indices.append(peak)

    while True:
        diffs = np.diff(p2_motion_indices)
        to_del = []
        for i, diff in enumerate(diffs):
            if diff < 40:
                max_in = np.argmax([dists[p2_motion_indices[i]], dists[p2_motion_indices[i + 1]]])
                to_del.append(i + max_in)

        p2_motion_indices = np.delete(p2_motion_indices, to_del)
        if len(to_del) == 0:
            break

    return p2_motion_indices