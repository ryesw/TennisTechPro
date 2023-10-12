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
        self.seq_length = 45 # 최근의 45개의 동작 데이터를 보고 다음 동작을 예측
        self.columns = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
            'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle'
        ]
        self.class_names = {'backhand2hands': '2BH', 'backhand': 'BHD', 'backhand_slice': 'BSL', 'backhand_volley': 'BVY', 
                            'forehand_flat': 'FFT', 'forehand_openstands': 'FOS', 'forehand_slice': 'FSL', 'forehand_volley': 'FVY', 
                            'flat_service': 'SFT', 'kick_service': 'SKK', 'slice_service': 'SSL', 'smash': 'SMS'}
        self.classes = {'backhand2hands': 0, 'backhand': 1, 'backhand_slice': 2, 'backhand_volley': 3, 
                        'forehand_flat': 4, 'forehand_openstands': 5, 'forehand_slice': 6, 'forehand_volley': 7,  
                        'flat_service': 8, 'kick_service': 9, 'slice_service': 10, 'smash': 11}
        self.keypoints_lists = []
        self.features_len = features_len

    
    def save_video_paths_to_csv(self):
        """
        Save the path of videos as csv file"""
        base_path = "THETIS\VIDEO_RGB"

        data = []
        for motion in os.listdir(base_path):
            motion_path = os.path.join(base_path, motion)
            
            if os.path.isdir(motion_path):
                for video_name in os.listdir(motion_path):
                    video_path = os.path.join(motion_path, video_name)
                    data.append([motion, video_name, video_path])
                        
        df = pd.DataFrame(data, columns=["motion", "video_name", "video_path"])
        df.to_csv("video_paths.csv", index=False)


    def collect_datasets(self, motion):
        csv_file_path = 'video_paths.csv'
        df = pd.read_csv(csv_file_path)
        output_path = f"keypoints/{motion}.csv"

        for idx, row in df.iterrows():
            video_path = row['video_path']

            if motion == row['motion']:
                self.save_keypoints(video_path)
    
        self.save_keypoints_to_csv(output_path) # 하나의 motion에 대한 keypoint 리스트를 csv 파일에 저장
        self.keypoints_lists = [] # 다른 motion의 keypoint를 저장하기 위해 list 초기화

        print(f"{motion} finish\n")


    def save_keypoints(self, video_path):
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            print(f"Error: Could not open video '{video_path}'")
            return
        
        while True:
            ret, frame = video.read()

            if not ret:
                break

            keypoints = self.calculate_keypoints(frame) # YOLO 모델 적용해서 keypoints 좌표 계산
            self.keypoints_lists.append(keypoints) # List에 keypoint 저장
        
        video.release()
        # cv2.destroyAllWindows()
    

    def calculate_keypoints(self, image):
        results = thetis_model.predict(image)

        for result in results:
            kpts = result.keypoints.xy[0].cpu().numpy() # 17 x 2 배열
            kpts = kpts.flatten() # [x1, y1, x2, y2, ...] -> 34 x 1 배열

            # 테니스 동작을 하는 사람에 대해서만 keypoints 좌표를 얻을 수 있도록 수정 필요

        return kpts
    
    
    def save_keypoints_to_csv(self, output_path):
        columns = [f'{part}_{coord}' for part in self.columns for coord in ['x', 'y']]
        df = pd.DataFrame(self.keypoints_lists, columns=columns)
        df.to_csv(output_path, index=False)


def create_train_valid_test_datasets(csv_file, root_dir, transform=None):
    """
    Split Thetis dataset into train validation and test sets
    """
    videos_name = pd.read_csv(csv_file)
    test_player_id = 40
    test_videos_name = videos_name[
        videos_name.loc[:, 'name'].str.contains(f'p{test_player_id}', na=False)]
    remaining_ids = list(range(1, 55))
    remaining_ids.remove(test_player_id)
    valid_ids = np.random.choice(remaining_ids, 5, replace=False)
    mask = videos_name.loc[:, 'name'].str.contains('|'.join([f'p{id}' for id in valid_ids]), na=False)
    valid_videos_name = videos_name[mask]
    train_videos = videos_name.drop(index=test_videos_name.index.union(valid_videos_name.index))
    train_ds = ThetisDataset(train_videos, root_dir, transform=transform)
    valid_ds = ThetisDataset(valid_videos_name, root_dir, transform=transform)
    test_ds = ThetisDataset(test_videos_name, root_dir, transform=transform)
    return train_ds, valid_ds, test_ds

def get_dataloaders(csv_file, root_dir, transform, batch_size, dataset_type='stroke', num_classes=256, num_workers=0, seed=42):
    """
    Get train and validation dataloader for strokes and tracknet datasets
    """
    ds = []
    if dataset_type == 'stroke':
        ds = StrokesDataset(csv_file=csv_file, root_dir=root_dir, transform=transform, train=True, use_features=True)
    elif dataset_type == 'tracknet':
        ds = TrackNetDataset(csv_file=csv_file, train=True, num_classes=num_classes)
    length = len(ds)
    train_size = int(0.85 * length)
    train_ds, valid_ds = torch.utils.data.random_split(ds, (train_size, length - train_size),
                                                       generator=torch.Generator().manual_seed(seed))
    print(f'train set size is : {train_size}')
    print(f'validation set size is : {length - train_size}')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_dl, valid_dl

if __name__ == '__main__':
    thetis = ThetisDataset()

    motions = thetis.class_names
    for motion in motions.keys():
        thetis.collect_datasets(motion)