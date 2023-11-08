from ultralytics import YOLO

detection_model = YOLO('yolo/clip.pt')
pose_model = YOLO('yolo/yolov8l-pose.pt')
thetis_model = YOLO('yolo/yolov8l-pose.pt')