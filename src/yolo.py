from ultralytics import YOLO

detection_model = YOLO('yolo/best_detection.pt')
# detection_model.train(data='config.yaml', epochs=10, batch=8, workers=1, device=0)
# detection model train 계속 진행해야 함

pose_model = YOLO('yolo/yolov8m-pose.pt')
# pose_model.train(data='pose_config.yaml', epochs=10, batch=8, workers=1, device=0)
# Pose model도 train 진행해야 함
# yaml은 pose estimation을 위해 따로 만들어야 함
# results = pose_model(source="test/video_input1.mp4", show=True, conf=0.4, save=True)