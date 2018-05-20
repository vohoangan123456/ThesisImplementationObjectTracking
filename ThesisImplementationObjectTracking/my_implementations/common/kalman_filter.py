'''
    File name         : kalman_filter.py
    File Description  : handle the kalman filter method
    Author            : An Vo
    Date created      : 19/04/2018
    Python Version    : 3.6
'''
import numpy as np
from filterpy.kalman import KalmanFilter
from my_implementations.common.bounding_box import BoundingBox
def convert_bbox_to_standard(bounding_box):
    '''
        Description:
            convert bounding box to the form that kalman filter can use
        Params:
            bounding_box(BoundingBox): the bounding box of object
        Return:
            the standard form of bounding box that kalman filter can use
    '''
    width = bounding_box.width
    height = bounding_box.height
    pX = bounding_box.pX + width / 2.
    pY = bounding_box.pY + height / 2.
    area = width * height         # area of bounding box
    if height == 0:
        return np.array([pX, pY, 0, 0]).reshape((4,1))
    ratio = width / float(height) # ratio between width and height
    return np.array([pX, pY, area, ratio]).reshape((4,1))

def convert_standard_to_bbox(standard):
    '''
        Description:
            convert standard format that get from the prediction of kalman filter to bounding box
        Params:
            standard([pX, pY, area, ratio]): the standard format data of kalman filter
        Return:(BoundingBox)
            - the bounding box
    '''
    width = np.sqrt(standard[2]*standard[3])
    if width == 0:
        return BoundingBox(0, 0, 0, 0)
    height = standard[2]/width
    bounding_box = BoundingBox(int(standard[0][0]-width[0]/2), int(standard[1][0]-height[0]/2), int(width[0]), int(height[0]))
    return bounding_box

class KalmanFilterTracker(object):
    count = 0
    def __init__(self,bbox):
        '''
            Description:
                Initialize tracker with initial bounding box
            Params:
                bbox(BoundingBox): bounding box object need to be tracked
        '''
        # define constant for kalman filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_standard(bbox)
        #self.time_since_update = 0
        self.id = KalmanFilterTracker.count
        KalmanFilterTracker.count += 1
        #self.hits = 0
        #self.hit_streak = 0
        #self.age = 0
        self.trace = []

    def get_current_state(self):
        '''
            Description:
                Get the current state of the bounding box
        '''
        return convert_standard_to_bbox(self.kf.x)

    def update(self,bbox):
        '''
            Description:
                Update the state of the bounding box with the actual bounding box
            Params:
                bbox(BoundingBox): the observed bounding box
        '''
        #self.time_since_update = 0
        #self.hits += 1
        #self.hit_streak += 1
        self.trace = []
        self.kf.update(convert_bbox_to_standard(bbox))

    def predict(self):
        '''
            Description:
                Predict the next state of the current bounding box
        '''
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        #self.age += 1
        #if(self.time_since_update>0):
        #    self.hit_streak = 0
        #self.time_since_update += 1
        self.trace.append(convert_standard_to_bbox(self.kf.x))
        return self.trace[-1]