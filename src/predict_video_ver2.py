import cv2
import imutils
import time

from court_detection import CourtDetector
from object_tracking import Tracker
from pose import PoseExtractor
from action_recognition import ActionRecognition
from ball_detection_ver2 import BallDetector
from utils import get_video_properties

def merge(frame, image):
    frame_h, frame_w = frame.shape[:2]

    width = frame_w // 4
    resized = imutils.resize(image, width=width)

    img_h, img_w = resized.shape[:2]
    w = frame_w - img_w

    frame[:img_h, w:] = resized

    return frame

def create_minimap(court_detector, tracker, ball_detector, fps, p1_predictions, p2_predictions):
    """
    Creates top view video of the gameplay
    """
    court = court_detector.court_reference.court.copy()
    court = cv2.line(court, *court_detector.court_reference.net, 255, 5)
    v_width, v_height = court.shape[::-1]
    court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output/minimap.mp4', fourcc, fps, (v_width, v_height))

    # Marking players location on court
    frame_num = 0
    p1_frame, p2_frame = -11, -11
    smoothed_1, smoothed_2 = tracker.calculate_feet_positions(court_detector) # 선수들의 발 좌표 계산
    for feet_pos_1, feet_pos_2 in zip(smoothed_1, smoothed_2):
        frame = court.copy()
        if feet_pos_1[0] is not None:
            frame = cv2.circle(frame, (int(feet_pos_1[0]), int(feet_pos_1[1])), 30, (255, 0, 255), -1) # 선수 1의 발 좌표를 미니맵에 표시
            
            # 선수 1의 동작을 미니맵에 표시
            if frame_num + 10 in p1_predictions.keys():
                p1_frame = frame_num + 10
                p1_motion = p1_predictions[p1_frame]['motion']

            if frame_num <= p1_frame + 10:
                frame = cv2.putText(frame, p1_motion, (int(feet_pos_1[0] + 25), int(feet_pos_1[1]) + 20), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 255), 4)
        
        if feet_pos_2[0] is not None:
            frame = cv2.circle(frame, (int(feet_pos_2[0]), int(feet_pos_2[1])), 30, (255, 255, 0), -1) # 선수 2의 발 좌표를 미니맵에 표시

            # 선수 2의 동작을 미니맵에 표시
            if frame_num + 10 in p2_predictions.keys():
                p2_frame = frame_num + 10
                p2_motion = p2_predictions[p2_frame]['motion']

            if frame_num <= p2_frame + 10:
                frame = cv2.putText(frame, p2_motion, (int(feet_pos_2[0] + 25), int(feet_pos_2[1]) - 20), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 0), 4)
        
        frame = ball_detector.draw_ball_position_in_minimap(frame, court_detector, frame_num) # ball의 좌표를 미니맵에 표시
        frame_num += 1
        out.write(frame)
    out.release()

def add_minimap(output_video_path):
    video = cv2.VideoCapture('output/analysis.mp4')
    fps, length, v_width, v_height = get_video_properties(video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (v_width, v_height))
    
    print('Adding the minimap...')

    minimap_video = cv2.VideoCapture('output/minimap.mp4')
    while True:
        ret, frame = video.read()
        ret2, minimap_frame = minimap_video.read()
        if ret and ret2:
            output = merge(frame, minimap_frame)
            output_video.write(output)
        else:
            break
    video.release()
    minimap_video.release()
    output_video.release()


def process(input_video_path, output_video_path):
    # Set start time
    start_time = time.time()

    # Load videos from videos path
    video = cv2.VideoCapture(input_video_path)

    # Get videos properties
    fps, length, v_width, v_height = get_video_properties(video)

    # Output Video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    analysis_video = cv2.VideoWriter('output/analysis.mp4', fourcc, fps, (v_width, v_height))

    # Initialize
    court_detector = CourtDetector() # Court Detector
    tracker = Tracker() # Player tracker
    pose_extractor = PoseExtractor() # Pose Extractor
    action_recognition = ActionRecognition() # Action Recognition
    ball_detector = BallDetector() # tracknet

    frame_num = 0 # frame counter
    frames = [] # Save all frame


    # First Part: Object Tracking, Ball Detection
    while True:
        ret, frame = video.read()
        if frame_num == 0:
            print('Start!')
            print('Detecting and Tracking the ball and players... ')
        if ret:
            # Tracking two players
            tracker.track(frame, frame_num)

            # Detect ball
            ball_detector.detect_ball(frame)

        else:
            break
        frame_num += 1
    video.release() # Video 획득 개체를 해제


    # Second Part: Player Detection, Court Detection, Pose Estimation
    tracker.find_players_boxes()

    video = cv2.VideoCapture(input_video_path)
    frame_num = 0

    while True:
        ret, frame = video.read()
        if ret:
            if frame_num == 0:
                print('Detecting the court...')
                lines = court_detector.detect(frame)
            else:
                # then track court
                lines = court_detector.track_court(frame)
            
            # Draw boxes on player's position
            frame, p1_boxes, p2_boxes = tracker.mark_boxes(frame, frame_num)

            # Pose Estimation
            frame = pose_extractor.extract_pose(frame, p1_boxes, p2_boxes)

            # Draw court lines
            new_frame = court_detector.draw_court_lines(frame, lines)
            frames.append(new_frame)

            analysis_video.write(new_frame)
        else:
            break
        frame_num += 1

    video.release() # Video 획득 개체를 해제
    analysis_video.release() # Video 획득 개체를 해제

    # Third Part: Processing ball coordinates
    print('전처리 전 \n', ball_detector.xy_coordinates)
    ball_detector.preprocessing_ball_coords()
    print('전처리 후 \n', ball_detector.xy_coordinates)

    # Fourth Part: Action Recognition
    p1_df, p2_df = pose_extractor.save_to_csv() # 선수들의 pose를 추정한 좌표를 저장
    p1_predictions, p2_predictions = action_recognition.predict(tracker, p1_df, p2_df, ball_detector.xy_coordinates, frame_num)
    print('p1_motions\n', p1_predictions)
    print('p2_motions\n', p2_predictions)

    # Fifth Part: Add minimap in video
    create_minimap(court_detector, tracker, ball_detector, fps, p1_predictions, p2_predictions) # minimap video를 생성
    add_minimap(output_video_path) # output video와 minimap video를 합친 하나의 video 생성
    
    # Measure processing time
    end_time = time.time()
    total_time = end_time - start_time

    print('Finished!')
    print(f'Analysis Time: {round(total_time)}s')
    print('Total Frame: ', frame_num)
    tracker.print_counts()
    pose_extractor.print_counts()

if __name__ == '__main__':
    input_video_path = 'test/video_input1.mp4'
    output_video_path = 'output/output.mp4'
    process(input_video_path, output_video_path)