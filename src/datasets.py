import torch
import os
import cv2
import numpy as np
import pandas as pd

from yolo import thetis_model
from utils import get_video_properties

class ThetisDataset:
    def __init__(self, csv_file=None, root_dir=None, transform=None, train=True, use_features=True, class_names=True,
                 features_len=100):
        self.videos_name = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.use_features = use_features
        self.seq_length = 0 # 최근의 n개의 동작 데이터를 보고 다음 동작을 예측, FPS로 설정
        self.columns = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 
            'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        self.class_names = {'backhand2hands': '2BH', 'backhand': 'BHD', 'backhand_slice': 'BSL', 'backhand_volley': 'BVY', 
                            'forehand_flat': 'FFT', 'forehand_openstands': 'FOS', 'forehand_slice': 'FSL', 'forehand_volley': 'FVY', 
                            'flat_service': 'SFT', 'kick_service': 'SKK', 'slice_service': 'SSL', 'smash': 'SMS'}
        self.classes = {'backhand2hands': 0, 'backhand': 1, 'backhand_slice': 2, 'backhand_volley': 3, 
                        'forehand_flat': 4, 'forehand_openstands': 5, 'forehand_slice': 6, 'forehand_volley': 7,  
                        'flat_service': 8, 'kick_service': 9, 'slice_service': 10, 'smash': 11}
        self.keypoints_list = []
        self.features_len = features_len


    def collect_datasets(self, motion):
        """
        Collect data for a specific motion from a CSV file containing video paths
        and Calculate keypoints for each frame in the videos and saves them in a CSV file.
        """
        csv_file_path = 'csv/extract_video_paths.csv'
        df = pd.read_csv(csv_file_path)

        output_path = f"keypoints/{motion}.csv" # Keypoints를 저장할 CSV 파일

        motion_video_paths = df[df['motion'] == motion] # 특정 Motion에 대한 Video paths 추출
        class_idx = np.array([self.classes[motion]], dtype=np.float32) # Motion의 Label Number

        for idx, row in motion_video_paths.iterrows():
            video_path = row['video_path']

            self.save_keypoints(video_path, class_idx) # 52 x 1 배열

        self.save_keypoints_to_csv(output_path) # 하나의 motion에 대한 keypoint 리스트를 csv 파일에 저장

        self.keypoints_list = [] # 한 Motion에 대한 좌표를 모두 저장했으면 다음 Motion을 위해 초기화


    def save_keypoints(self, video_path, class_idx):
        """
        Process a video to calculate keypoints and appends them to a list
        """
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            print(f"Error: Could not open video '{video_path}'")
            return
        
        fps, _, _, _ = get_video_properties(video)
        self.seq_length = int(fps)
        
        while True:
            ret, frame = video.read()

            if not ret:
                break

            keypoints_xyv = self.calculate_keypoints(frame) # YOLO 모델 적용해서 keypoints 좌표 계산

            data = np.append(keypoints_xyv, class_idx)
            self.keypoints_list.append(data)
    

    def calculate_keypoints(self, image):
        """
        Calculate keypoints for a frame using a YOLO pose model
        """
        results = thetis_model.predict(image)

        for result in results:
            kpts = result.keypoints.data[0].cpu().numpy() # 17 x 3 배열
            keypoints_xyv = kpts.flatten() # [x1, y1, v1, x2, y2, v2, ...] -> 51 x 1 배열

        return keypoints_xyv
    
    
    def save_keypoints_to_csv(self, output_path):
        """
        Save the collected keypoints to a CSV file
        """
        columns = [f'{motion}_{coord}' for motion in self.columns for coord in ['x', 'y', 'v']] + ['class']
        df = pd.DataFrame(self.keypoints_list, columns=columns)
        df.to_csv(output_path, index=False)

    
    def create_sequence_dataset(self):

        x_train, y_train, x_valid, y_valid, x_test, y_test = split_train_valid_test_datasets()
        
        x_train_sequence, y_train_sequence = self.convert_sequence_dataset(x_train, y_train)
        x_valid_sequence, y_valid_sequence = self.convert_sequence_dataset(x_valid, y_valid)
        x_test_sequence, y_test_sequence = self.convert_sequence_dataset(x_test, y_test)

        return x_train_sequence, y_train_sequence, x_valid_sequence, y_valid_sequence, x_test_sequence, y_test_sequence
    
    
    def convert_sequence_dataset(self, x_data, y_data):
        x_dataset = []
        y_dataset = []

        for i in range(len(x_data) - self.seq_length):
            _x = x_data[i:i + self.seq_length]
            x_dataset.append(_x)
            _y = y_data[i] 
            y_dataset.append(_y)

        return np.array(x_dataset), np.array(y_dataset)
    

def save_video_paths_to_csv():
    """
    Save the path of videos as CSV file
    """
    base_path = "THETIS\VIDEO_RGB"

    data = []
    for motion in os.listdir(base_path):
        motion_path = os.path.join(base_path, motion)
        
        if os.path.isdir(motion_path):
            for video_name in os.listdir(motion_path):
                video_path = os.path.join(motion_path, video_name)
                data.append([motion, video_name, video_path])
                    
    df = pd.DataFrame(data, columns=["motion", "video_name", "video_path"])
    df.to_csv("csv/total_video_paths.csv", index=False)


def separate_video_paths():
    """
    Separate video paths
    """
    video_paths = 'csv/total_video_paths.csv'
    df = pd.read_csv(video_paths)

    extract_video_paths_list = []
    remaining_video_paths_list = []
    motions = ['backhand2hands', 'backhand', 'backhand_slice', 'backhand_volley', 
               'forehand_flat', 'forehand_openstands', 'forehand_slice', 'forehand_volley', 
               'flat_service', 'kick_service', 'slice_service', 'smash']
    
    for motion in motions:
        motion_video_data = df[df['motion'] == motion]

        for idx, row in motion_video_data.iterrows():
            video_name = row['video_name']
            video_path = row['video_path']

            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                print(f"Error: Could not open video '{video_path}'")
                break

            pose_count = 0
            while pose_count == 0:
                ret, frame = video.read()
                if not ret:
                    break

                results = thetis_model.predict(frame)
                for result in results:
                    kpts = result.keypoints.xy
                    if len(kpts) >= 2:
                        pose_count += 1
            
            if pose_count == 0:
                extract_video_paths_list.append([motion, video_name, video_path])
            else:
                remaining_video_paths_list.append([motion, video_name, video_path])

    train_path_df = pd.DataFrame(extract_video_paths_list, columns=["motion", "video_name", "video_path"])
    train_path_df.to_csv('csv/extract_video_paths.csv', index=False)

    valid_path_df = pd.DataFrame(remaining_video_paths_list, columns=["motion", "video_name", "video_path"])
    valid_path_df.to_csv('csv/remaining_video_paths.csv', index=False)


def split_datasets(df):
    length = len(df)
    a = int(length * 0.8)
    b = int((length - a) / 2 + 1)

    train = df[:a]
    valid = df[a:a+b]
    test = df[a+b:]

    return train, valid, test


def split_train_valid_test_datasets():
    """
    Split THETIS Dataset into train, validation and test sets
    """
    base_path = 'keypoints/'

    train_dfs = []
    valid_dfs = []
    test_dfs = []
    
    # 하나의 동작에 대한 csv 파일에서 train/valid/test로 나눔
    for csv_file in os.listdir(base_path):
        df = pd.read_csv(base_path + csv_file)

        train, valid, test = split_datasets(df)

        train_dfs.append(train)
        valid_dfs.append(valid)
        test_dfs.append(test)

    train_df = pd.concat(train_dfs, ignore_index=True)
    valid_df = pd.concat(valid_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    x_train = train_df.drop('class', axis=1).values
    y_train = train_df['class'].reset_index(drop=True).values

    x_valid = valid_df.drop('class', axis=1).values
    y_valid = valid_df['class'].reset_index(drop=True).values

    x_test = test_df.drop('class', axis=1).values
    y_test = test_df['class'].reset_index(drop=True).values

    return x_train, y_train, x_valid, y_valid, x_test, y_test



# def get_dataloaders(csv_file, root_dir, transform, batch_size, dataset_type='stroke', num_classes=256, num_workers=0, seed=42):
#     """
#     Get train and validation dataloader for strokes and tracknet datasets
#     """
#     ds = []
#     if dataset_type == 'stroke':
#         ds = StrokesDataset(csv_file=csv_file, root_dir=root_dir, transform=transform, train=True, use_features=True)
#     elif dataset_type == 'tracknet':
#         ds = TrackNetDataset(csv_file=csv_file, train=True, num_classes=num_classes)
#     length = len(ds)
#     train_size = int(0.85 * length)
#     train_ds, valid_ds = torch.utils.data.random_split(ds, (train_size, length - train_size),
#                                                        generator=torch.Generator().manual_seed(seed))
#     print(f'train set size is : {train_size}')
#     print(f'validation set size is : {length - train_size}')

#     train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     return train_dl, valid_dl


if __name__ == '__main__':
    thetis = ThetisDataset()

    x_train, y_train, x_valid, y_valid, x_test, y_test = thetis.create_sequence_dataset()

    print(x_train.shape)
    print(y_train.shape)

    print(x_valid.shape)
    print(y_valid.shape)

    print(x_test.shape)
    print(y_test.shape)