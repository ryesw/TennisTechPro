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
input_video_number = str(1)
video_path = 'test/video_input' + input_video_number + '.mp4'
video = cv2.VideoCapture(video_path)

# Get videos properties
fps, length, v_width, v_height = get_video_properties(video)

# Ouput Video
output_video_number = '456'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = 'output/output' + output_video_number + '.mp4'
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (v_width, v_height))

dtype = get_dtype()

# Court Detector
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
            print('Start!')
            print('Detecting the court and the players...')
            print("Estimate player's pose...")
            lines = court_detector.detect(frame)
        else:
            # then track court
            lines = court_detector.track_court(frame)
        
        # Detect two players
        frame = player_detector.detect(frame) # frame vs frame.copy()

        # Estimate player's pose
        player1_boxes, player2_boxes = player_detector.get_boxes()
        frame = pose_extractor.extract_pose(frame, player1_boxes, player2_boxes)

        # Player Box 초기화
        player_detector.initialize_box()
        
        # # Detect ball
        # ball_detector.detect_ball(court_detector.delete_extra_parts(frame))

        new_frame = court_detector.draw_court_lines(frame, lines)
        frames.append(new_frame)
            
        output_video.write(new_frame)
    else:
        break
output_video.release()
print('Finished!')

print('Total Frame: ', frame_i)
player_detector.print_counts()
pose_extractor.print_counts()

# ball track code