import argparse
import cv2

from court_detection import CourtDetector
from detection import DetectionModel
from pose import PoseExtractor
from utils import get_dtype, get_video_properties

# parse parameters
parser = argparse.ArgumentParser(description='Tennis Game Analysis')

parser.add_argument("--input_video_path", type=str)
parser.add_argument("--output_video_path", type=str, default="")
parser.add_argument("--stickman", type=int, default=0)
parser.add_argument("--minimap", type=int, default=0)
parser.add_argument("--bounce", type=int, default=0)

args = parser.parse_args()
input_video_path = args.input_video_path
output_video_path = args.output_video_path
minimap = args.minimap
bounce = args.bounce



# Load videos from videos path
video_path = 'test/video_input1.mp4'
video = cv2.VideoCapture(video_path)

# get videos properties
fps, length, v_width, v_height = get_video_properties(video)

# Ouput Video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = 'output/output1.mp4'
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (v_width, v_height))

dtype = get_dtype()

# Court
court_detector = CourtDetector()

# Players tracker
player_detector = DetectionModel()

# Pose Extractor
pose_extractor = PoseExtractor()

# # Motion Recognition
# stroke_recognition = ActionRecognition('storke_classifier_weights.pth')

# # Ball Detector
# ball_detector = BallDetector('saved states/tracknet_weights_2_classes.pth', out_channels=2)

coords = []
frame_i = 0 # frame counter
frames = []
t = []


# First Part
while True:
    ret, frame = video.read()
    frame_i += 1
    if ret:
        if frame_i == 1:
            print('start')
            print('Detecting the court and the players...')
            lines = court_detector.detect(frame)
        else:
            # then track court
            lines = court_detector.track_court(frame)
        
        # detect
        frame = player_detector.detect(frame.copy()) # frame vs frame.copy()

        player1_boxes = player_detector.player_1_boxes
        player2_boxes = player_detector.player_2_boxes

        # pose estimation
        frame = pose_extractor.extract_pose(frame, player1_boxes, player2_boxes)
        
        # # ball detect
        # ball_detector.detect_ball(court_detector.delete_extra_parts(frame))
        
        # 한 frame에서 court에 선을 그리는 함수
        for i in range(0, len(lines), 4):
            x1, y1, x2, y2 = lines[i], lines[i+1], lines[i+2], lines[i+3]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 5)
        new_frame = cv2.resize(frame, (v_width, v_height))
        frames.append(new_frame)
            
        output_video.write(new_frame)
    else:
        break
output_video.release()
print('Finished!')

# ball track code