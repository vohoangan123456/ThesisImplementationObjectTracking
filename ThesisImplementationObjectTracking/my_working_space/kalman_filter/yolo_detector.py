import cv2
from my_working_space.select_object import BoundingBox
class yolo_detector:
    def __init__(self, tfNet):
        self.tfNet = tfNet
    def detect(self, frame):
        results = self.tfNet.return_predict(frame)
        centers = []
        for result in results:
            label = result['label']
            if label == 'person':
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                [x,y] = [tl[0], tl[1]]
                bounding_box = BoundingBox(tl[0], tl[1], abs(tl[0] - br[0]), abs(tl[1] - br[1]))
                cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
                b = np.array([[x], [y]])
                centers.append(np.round(b))
        return centers
                
