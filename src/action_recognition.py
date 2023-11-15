import numpy as np
from scipy.signal import find_peaks

from utils import center_of_box

from tensorflow.keras.models import load_model


class ActionRecognition:
    def __init__(self):
        self.model = load_model('models/gru/gru64masking.h5')
        self.seq_length = 60
        self.motions = ['backhand', 'forehand', 'service', 'smash', 'backhand_volley', 'forehand_volley']
        self.player1_predictions = {}
        self.player2_predictions = {}

    def predict(self, tracker, p1_keypoints_df, p2_keypoints_df, ball_positions, total_frame):
        p1_boxes = tracker.player1_boxes
        p2_boxes = tracker.player2_boxes
        p1_first_appearance_frame = tracker.p1_first_appearance_frame
        p2_first_appearance_frame = tracker.p2_first_appearance_frame

        # bottom 선수와 top 선수의 box 좌표를 구분
        # if p1_boxes[p1_first_appearance_frame][1] < p2_boxes[p2_first_appearance_frame][1]:
        #     swap = p1_boxes
        #     p1_boxes = p2_boxes
        #     p2_boxes = swap

        p1_motion_frames = find_p1_motion_frame(p1_boxes, ball_positions, p1_keypoints_df)
        p2_motion_frames = find_p2_motion_frame(p2_boxes, ball_positions, p2_keypoints_df)

        for frame_num in p1_motion_frames:
            probs, motion = self.predict_p1_motion(frame_num, p1_keypoints_df, total_frame)
            self.player1_predictions[frame_num] = {'probs': probs, 'motion': motion}
            
        for frame_num in p2_motion_frames:
            probs, motion = self.predict_p2_motion(frame_num, p2_keypoints_df, total_frame)
            self.player2_predictions[frame_num] = {'probs': probs, 'motion': motion}

        return self.player1_predictions, self.player2_predictions

    def predict_p1_motion(self, frame_num, p1_keypoints_df, total_frame):
        p1_keypoints_df = p1_keypoints_df.drop(p1_keypoints_df.columns[2:10], axis=1) # eye와 ear에 관련된 column 제거
        p1_keypoints = p1_keypoints_df.values

        # bottom player의 좌표를 좌우 대칭
        # p1_keypoints = np.array([[1 - value if index % 2 == 0 else value for index, value in enumerate(inner_list)] for inner_list in p1_keypoints])

        mid = int(self.seq_length // 2)
        before_row = max(0, frame_num - mid - 1)
        after_row = min(before_row + self.seq_length, total_frame)
        
        if before_row == 0:
            kpts = p1_keypoints[before_row : before_row + self.seq_length]
        elif after_row == total_frame:
            kpts = p1_keypoints[after_row - self.seq_length : after_row]
        else:
            kpts = p1_keypoints[before_row : after_row]

        kpts = np.array(kpts).reshape(1, self.seq_length, 26)
        probs = self.model.predict(kpts)[0]
        idx = np.argmax(probs)

        return probs[idx], self.motions[idx]

    def predict_p2_motion(self, frame_num, p2_keypoints_df, total_frame):
        p2_keypoints_df = p2_keypoints_df.drop(p2_keypoints_df.columns[2:10], axis=1) # eye와 ear에 관련된 column 제거
        p2_keypoints = p2_keypoints_df.values

        mid = int(self.seq_length // 2)
        before_row = max(0, frame_num - mid - 1)
        after_row = min(before_row + self.seq_length, total_frame)

        if before_row == 0:
            kpts = p2_keypoints[before_row : before_row + self.seq_length]
        elif after_row == total_frame:
            kpts = p2_keypoints[after_row - self.seq_length : after_row]
        else:
            kpts = p2_keypoints[before_row : after_row]
            
        kpts = np.array(kpts).reshape(1, self.seq_length, 26)
        probs = self.model.predict(kpts)[0]
        idx = np.argmax(probs)

        return probs[idx], self.motions[idx]


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
    print('p1 peaks: ', peaks)
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

    print('p1 motion indices: ', p1_motion_indices)
    return p1_motion_indices

def find_p2_motion_frame(p2_boxes, ball_positions, p2_keypoints_df):
    left_wrist_index = 18
    right_wrist_index = 20
    left_wrist_pos = p2_keypoints_df.iloc[:, [left_wrist_index, left_wrist_index+1]].values
    right_wrist_pos = p2_keypoints_df.iloc[:, [right_wrist_index, right_wrist_index+1]].values

    peaks, _ = find_peaks(ball_positions[:, 1] * (-1))
    print('p2 peaks: ', peaks)
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

    print('p2 motion indices: ', p2_motion_indices)
    return p2_motion_indices