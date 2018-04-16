import cv2
import numpy as np
from my_working_space.kalman_filter.moving_object import MovingObject, BoundingBox

class yolo_detector:
    def __init__(self, tfNet):
        self.tfNet = tfNet
        self.list_moving_obj = []
    def detect(self, frame):
        results = self.tfNet.return_predict(frame)
        self.list_moving_obj = []
        for result in results:
            label = result['label']
            confidence = result['confidence']
            if label == 'person':
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])                
                bounding_box = BoundingBox(tl[0], tl[1], abs(tl[0] - br[0]), abs(tl[1] - br[1]))
                moving_obj = MovingObject(frame, bounding_box)
                self.list_moving_obj.append(moving_obj);
                cv2.rectangle(frame, tl, br, (0, 255, 0), 2)                
                
