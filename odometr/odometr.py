import cv2 as cv
import pandas as pd
from ultralytics import YOLO

YOLO_DETECT = 'model_weights/odometr_detect_best.pt'

class Odometr:
    def __init__(self, yolo_detect_weights = YOLO_DETECT):
        self.y_d_weights = yolo_detect_weights
        
    def detection_model_result(self, img):
        model = YOLO(self.y_d_weights)
        result = model.predict(img, save_txt=False, save_crop=False)
        positions = pd.DataFrame(result[0].boxes.xyxy, columns=['upper_left_x','upper_left_y','lower_right_x','lower_right_y'])
        return positions
    
  