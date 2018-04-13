import cv2
import numpy as np

class Location:
    def __init__(self, pX, pY):
        self.pX = pX
        self.pY = pY

class BoundingBox:
    def __init__(self, pX, pY, width, height):
        self.pX = pX
        self.pY = pY
        self.width = width
        self.height = height
        self.center:Location = Location(pX + width / 2, pY + height / 2)

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
                bounding_box = BoundingBox(tl[0], tl[1], abs(tl[0] - br[0]), abs(tl[1] - br[1]))
                cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
                [x,y] = [bounding_box.center.pX, bounding_box.center.pY]
                b = np.array([[x], [y]])
                centers.append(np.round(b))
        return centers
                
