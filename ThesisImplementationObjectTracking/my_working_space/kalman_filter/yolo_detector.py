import cv2
import numpy as np
from my_working_space.kalman_filter.moving_object import MovingObject, BoundingBox

class yolo_detector:
    def __init__(self, tfNet):
        self.tfNet = tfNet
        self.list_moving_obj = []
        self.frame_index = 0
    def detect(self, frame):
        self.frame_index += 1
        cv2.putText(frame, str('frame: {0}'.format(self.frame_index)), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        print('--------------------------------', str(self.frame_index), '-----------------------------')
        print('# detection')
        results = self.tfNet.return_predict(frame)
        self.list_moving_obj = []
        for result in results:
            label = result['label']
            confidence = result['confidence']
            if label == 'person':
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])                
                if confidence >= 0.55:
                    bounding_box = BoundingBox(tl[0], tl[1], abs(tl[0] - br[0]), abs(tl[1] - br[1]))
                    moving_obj = MovingObject(frame, bounding_box)
                    moving_obj.get_feature()
                    self.list_moving_obj.append(moving_obj);
                    cv2.rectangle(frame, tl, br, (0, 255, 0), 1)
                    cv2.putText(frame, str('({0},{1})'.format(tl[0], tl[1])), (tl[0], br[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(frame, str("%.2f" % confidence), (br[0], tl[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                print(str("%.2f" % confidence), str('({0},{1})'.format(tl[0], tl[1])))
        # Slower the FPS
        cv2.waitKey(50)
                
