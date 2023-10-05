import torch
from yolo import detection_model
from ultralytics.utils.plotting import Annotator

class DetectionModel:
    def __init__(self, match_type, dtype=torch.FloatTensor):
        self.detection_model = detection_model
        self.match_type = match_type
        self.dtype = dtype
        self.PLAYER_1_LABEL = 0
        self.PLAYER_2_LABEL = 1
        self.PLAYER_3_LABEL = 2
        self.PLAYER_4_LABEL = 3
        self.RACKET_LABEL = 43
        self.BALL_LABEL = 37
        self.PERSON_SCORE_MIN = 0.85
        self.PERSON_SECONDARY_SCORE = 0.3
        self.RACKET_SCORE_MIN = 0.6
        self.BALL_SCORE_MIN = 0.6
        self.v_width = 0 # video_width
        self.v_height = 0 # video_height
        self.line_thickness = 3
        self.player_1_boxes = []
        self.player_2_boxes = []


    def detect(self, image, person_min_score=None):
        """
        Use YOLO Model to detect players in the one frame
        """
        results = self.detection_model.predict(image)
        
        for result in results:
            annotator = Annotator(image, line_width=self.line_thickness, font_size=None)

            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls.cpu().numpy()) # class 정보: bottom=0, top=1
                pts = box.xyxy[0].cpu().numpy() # box 좌표
                if cls == 0:
                    self.player_1_boxes.append(pts)
                if cls == 1:
                    self.player_2_boxes.append(pts)

                # annotator.box_label(pts, self.detection_model.names[cls]) -> Label까지 나오도록 함
                annotator.box_label(pts, label=False, color=(255, 0, 0))

        frame = annotator.result()
        return frame
    
def center_of_box(box):
    """
    Calculate the center of a box
    """
    if box[0] is None:
        return None, None
    height = box[3] - box[1]
    width = box[2] - box[0]
    return box[0] + width / 2, box[1] + height / 2