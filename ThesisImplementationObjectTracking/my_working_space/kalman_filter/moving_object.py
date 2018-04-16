from shapely.geometry import Point
import numpy as np

class BoundingBox:
    def __init__(self, pX, pY, width, height):
        self.pX = pX
        self.pY = pY
        self.width = width
        self.height = height
        self.center:Point = Point(pX + width / 2, pY + height / 2)

class MovingObject:
    def __init__(self, image, bounding_box:BoundingBox):
        self.bounding_box = bounding_box
        self.image = image[
                            self.bounding_box.pY:self.bounding_box.pY + self.bounding_box.height,
                            self.bounding_box.pX:self.bounding_box.pX + self.bounding_box.width
                        ]
        self.label = 0
        [x,y] = [bounding_box.center.x, bounding_box.center.y]
        self.center = np.round(np.array([[x], [y]]))

    def set_label(self, label:str):
        self.label = label
