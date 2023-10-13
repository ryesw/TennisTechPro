import torch
from yolo import detection_model
from ultralytics.utils.plotting import Annotator

class DetectionModel:
    def __init__(self, match_type=2, dtype=torch.FloatTensor):
        self.detection_model = detection_model
        self.match_type = match_type # 단식 2명, 복식 4명
        self.dtype = dtype
        self.BOTTOM_PLAYER_1 = 0
        self.BOTTOM_PLAYER_2 = 2
        self.TOP_PLAYER_1 = 1
        self.TOP_PLAYER_2 = 3
        self.BOTTOM_MIN_CONF = 0.55 # Bottom 선수 Detection 시 최소 confidence 값
        self.TOP_MIN_CONF = 0.32 # Top 선수 Detection 시 최소 confidence 값
        self.line_thickness = 2
        self.__player_1_boxes = []
        self.__player_2_boxes = []
        self.__player_1_count = 0
        self.__player_2_count = 0


    def detect(self, image, person_min_score=None):
        """
        Use YOLO Model to detect players in the one frame
        """
        results = self.detection_model.predict(image)

        for result in results:
            annotator = Annotator(image, line_width=self.line_thickness, font_size=None)

            boxes = result.boxes
            
            for i in range(len(boxes.xyxy)):
                box = boxes.xyxy[i].cpu().numpy()
                cls = boxes.cls[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                
                if cls == 0: # if cls == 0 and conf >= self.BOTTOM_MIN_CONF:
                    if len(self.__player_1_boxes) == 0:
                        self.__player_1_boxes.append(box)
                        self.__player_1_count += 1
                elif cls == 1: # elif cls == 1 and conf >= self.TOP_MIN_CONF:
                    if len(self.__player_2_boxes) == 0:
                        self.__player_2_boxes.append(box)
                        self.__player_2_count += 1

                annotator.box_label(box, label=False, color=(255, 0, 255))
            # 1. Detection Model의 정확도가 높을 때
            # for box in boxes:
            #     cls = int(box.cls.cpu().numpy()) # class 정보: bottom=0, top=1
            #     pts = box.xyxy[0].cpu().numpy() # box 좌표
            #     conf = box.conf[0].cpu().numpy() # confidence 값
                
            #     if cls == 0 and conf >= self.BOTTOM_MIN_CONF:
            #         self.__player_1_boxes.append(pts)
            #     elif cls == 1 and conf >= self.TOP_MIN_CONF:
            #         self.__player_2_boxes.append(pts)

            # 2. 최초의 2개만 
            # for idx, box in enumerate(boxes):
            #     cls = int(box.cls.cpu().numpy()) # class 정보: bottom=0, top=1
            #     pts = box.xyxy[0].cpu().numpy() # box 좌표

            #     if idx == 0:
            #         self.__player_1_boxes.append(pts)
            #     elif idx == 1:
            #         self.__player_2_boxes.append(pts)

                    # annotator.box_label(pts, self.detection_model.names[cls]) -> Label까지 나오도록 하는 Code
                    # annotator.box_label(box, label=False, color=(0, 0, 0))

        frame = annotator.result()
        return frame
    

    def initialize_box(self):
        self.__player_1_boxes = []
        self.__player_2_boxes = []


    def get_boxes(self):
        return self.__player_1_boxes, self.__player_2_boxes
    

    def print_counts(self):
        print('Player 1 Object Detection Count: ', self.__player_1_count)
        print('Player 2 Object Detection Count: ', self.__player_2_count)
    
    
def center_of_box(box):
    """
    Calculate the center of a box
    """
    if box[0] is None:
        return None, None
    height = box[3] - box[1]
    width = box[2] - box[0]
    return box[0] + width / 2, box[1] + height / 2