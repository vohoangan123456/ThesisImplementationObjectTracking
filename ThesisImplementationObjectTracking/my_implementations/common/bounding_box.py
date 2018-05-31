'''
    File name         : bounding_box.py
    File Description  : handle the bounding box object
    Author            : An Vo
    Date created      : 19/04/2018
    Python Version    : 3.6
'''
from my_implementations.common.global_config import *
from shapely.geometry import Point
import numpy as np
class BoundingBox:
    def __init__(self, pX, pY, width, height):
        '''
            Description:
                Initialize bounding box with top-left vertex and width height
            Params:
                pX(int): axis x of topleft vertex
                pY(int): axis y of topleft vertex
                width(int): width of the rectangle
                height(int): height of the rectangle
        '''
        self.pX = pX
        self.pY = pY
        self.pXmax = pX + width
        self.pYmax = pY + height
        self.width = width
        self.height = height
        self.center = Point(pX, pY + height)
        self.area = width * height
        self.is_under_of_occlusion = False  # flag check this bbx is under of another bbx
        self.is_topleft_occlusion = None    # flag check this bbx is under of another bbx and position is in topleft with another bbx
        self.is_disappear = False           # flag check this bbx is disappear (when the detector error)
        self.overlap_percent = 0            # percentage of overlapping

    def check_intersec_each_other(self, another_bbx):
        '''
            Description:
                check the whether this bbx intersect with another_bbx
            Params:
                another_bbx(BoundingBox): the another bbx that intersect with this bbx
            Returns: (bool)
                - True: two bounding box is intersected
                - False: two bounding box is not intersected
        '''
        # = delta x = min of top left x - max of bottom right x
        dx = min(self.pXmax, another_bbx.pXmax) - max(self.pX, another_bbx.pX)
        # = delta y = min of top left y - max of bottom right y
        dy = min(self.pYmax, another_bbx.pYmax) - max(self.pY, another_bbx.pY)
        if (dx>=0) and (dy>=0):
            return True
        return False

    def check_direction_of_intersec(self, another_bbx):
        '''
            Description:
                check the position top, left, right, bottom of this bbx to another_bbx
            params:
                another_bbx(BoundingBox): the another bbx that intersect with this bbx
            returns: (bool)
                - True: this rect is in top-left of another_bbx 
                - False: this rect is in top-right of another_bbx
        '''
        topLeft = min(self.pX, another_bbx.pX) == self.pX
        self.is_topleft_occlusion = topLeft
        return topLeft

    def check_behind_of_otherbbx(self, another_bbx):
        '''
            Description:
                check the whether this bbx is behind or in front of another_bbx
            Params:
                another_bbx(BoundingBox): the another bbx that intersect with this bbx
            Returns: bool
                True: this bbx is behind another_bbx
                False: this bbx is in front of another_bbx
        '''
        result = False
        if abs(self.pYmax - another_bbx.pYmax) > THRESHOLD_Y:
            # compare if y axis are much different
            maxY = max(self.pYmax, another_bbx.pYmax)
            result = maxY == another_bbx.pYmax
        else:
            # if two y axis are slightly different, we need to add bbx area criterion
            result = self.area < another_bbx.area
        #self.is_under_of_occlusion = result
        return result;

    def get_overlap_area(self, another_bbx):
        '''
            Description:
                get the percentage of overlaping of this bbx to another_bbx
            Params:
                another_bbx (BoundingBox): the another bbx that intersect with this bbx
            Returns: number
                - percentage of overlap area / total area
        '''
        xx1 = np.maximum(self.pX, another_bbx.pX)
        yy1 = np.maximum(self.pY, another_bbx.pY)
        xx2 = np.minimum(self.pXmax, another_bbx.pXmax)
        yy2 = np.minimum(self.pYmax, another_bbx.pYmax)
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        self.overlap_percent = int(wh / (self.width * self.height) * 100)
        return self.overlap_percent