import cv2
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
class CommonFOV:
    def __init__(self):
        self.list_point = []
        self.polygon = Polygon(self.list_point)

    def check_point_inside_FOV(self, point:Point):
        return self.polygon.contains(point)

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
        self.center:Point = Point(pX + width / 2, pY + height / 2)

class MovingObject:
    def __init__(self):
        self.bounding_box = None
        self.image = None        
        self.label = 0

    def create_object_with_boundingbox(self, image, bounding_box:BoundingBox):
        self.bounding_box = bounding_box
        self.image = image[
                            self.bounding_box.pY:self.bounding_box.pY + self.bounding_box.height,
                            self.bounding_box.pX:self.bounding_box.pX + self.bounding_box.width
                        ]
        self.label = 0

    def set_label(self, label:str):
        self.label = label

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
                
