from ultralytics import YOLO

detection_model = YOLO('yolo/best_detection.pt')
# detection_model = YOLO('yolo/best10.pt') -> 성능 좋지 않음
# detection_model = YOLO('yolo/best1-10.pt') -> 성능 좋지 않음
pose_model = YOLO('yolo/yolov8x-pose-p6.pt')
thetis_model = YOLO('yolo/yolov8x-pose-p6.pt')