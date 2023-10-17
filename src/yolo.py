from ultralytics import YOLO

detection_model = YOLO('yolo/best_detection.pt')
pose_model = YOLO('yolo/yolov8x-pose-p6.pt')
thetis_model = YOLO('yolo/yolov8x-pose-p6.pt')