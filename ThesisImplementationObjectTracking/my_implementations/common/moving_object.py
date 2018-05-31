'''
    File name         : moving_object.py
    File Description  : handle the moving object 
    Author            : An Vo
    Date created      : 19/04/2018
    Python Version    : 3.6
'''
import numpy as np
import copy
from sys import maxsize
from math import pow
from my_implementations.common.global_config import *
from my_implementations.common.features import FeatureMatching, FeatureExtractor
from my_implementations.common.bounding_box import BoundingBox

class MovingObject:
    def __init__(self, image, bounding_box):
        '''
            Description:
                Initialize moving object with the frame image and the bounding box of this moving object
            Params:
                image: the image get from camera frame
                bounding_box(BoundingBox): the bounding box of detected object
        '''
        self.bounding_box = bounding_box
        self.img_full = image
        self.image = image[
                            self.bounding_box.pY:self.bounding_box.pY + self.bounding_box.height,
                            self.bounding_box.pX:self.bounding_box.pX + self.bounding_box.width
                        ]
        self.label = 0
        [x,y] = [bounding_box.pX, bounding_box.pY]
        self.center = np.round(np.array([[x + int(bounding_box.width / 2)], [y + int(bounding_box.height / 2)]]))
        self.topLeft = np.round(np.array([[x], [y]]))
        self.topRight = np.round(np.array([[x + bounding_box.width], [y]]))
        self.bottomLeft = np.round(np.array([[x], [y + bounding_box.height]]))
        self.bottomRight = np.round(np.array([[x + bounding_box.width], [y + bounding_box.height]]))

        self.predict_bbx = bounding_box
        self.predict_image = image[
                            self.bounding_box.pY:self.bounding_box.pY + self.bounding_box.height,
                            self.bounding_box.pX:self.bounding_box.pX + self.bounding_box.width
                        ]
        self.predict_center = np.round(np.array([[x + int(bounding_box.width / 2)], [y + int(self.predict_bbx.height / 2)]]))
        self.predict_topLeft = np.round(np.array([[x], [y]]))
        self.predict_topRight = np.round(np.array([[x + self.predict_bbx.width], [y]]))
        self.predict_bottomLeft = np.round(np.array([[x], [y + self.predict_bbx.height]]))
        self.predict_bottomRight = np.round(np.array([[x + self.predict_bbx.width], [y + self.predict_bbx.height]]))
        
        self.HU_feature = None  # hu invariants
        self.CH_feature = None  # color histogram
        self.SI_feature = None  # sift
        self.vector = None      # vector from moving object to the nearest point in FOV
        self.is_in_fov = False
        self.confidence = 0
        self.list_distance = [] # list the distance from this obj to the edge of common fov
        self.exist_in = 0       # [0,1,2,3] <=> [not existed both, existed in cam1, existed in cam2, existed in both]

    def set_existed_in(self, camId):
        '''
            Description:
                set the flag to know where is this object existed
            Params:
                camId(int): the cameraId [1,2,3] <=> [cam1, cam2, both cam]
        '''
        self.exist_in += camId

    def set_label(self, label):
        '''
            Description:
                set label for moving object
            Params:
                label(str): label that need to set for object
        '''
        self.label = label

    def set_confidence(self, confidence):
        '''
            Description:
                set confidence of detection for moving object
            Params:
                confidence(float): confidence value that need to set for object
        '''
        self.confidence = confidence

    def set_vector(self, nearest_point):
        '''
            Description:
                set the vector from the moving object to the nearest point in the boundary of fov
            Params:
                nearest_point(Point): the nearest point to moving object
        '''
        self.vector = (self.bounding_box.center.x - nearest_point.x, self.bounding_box.center.y - nearest_point.y)

    def get_feature(self):
        '''
            Description:
                extract feature color histogram, hu invariants, sift of moving object
        '''
        feature_extractor = FeatureExtractor(self.image)
        self.CH_feature = feature_extractor.extract_color_histogram()
        self.HU_feature = feature_extractor.extract_hu_moment()
        self.SI_feature = feature_extractor.extract_sift_features()

    def update_bbx(self):
        '''
            Description:
                update the current bounding box of moving object by the prediction value
        '''
        self.bounding_box = copy.copy(self.predict_bbx)
        self.image = copy.copy(self.predict_image)
        self.center = copy.copy(self.predict_center)
        self.topLeft = copy.copy(self.predict_topLeft)
        self.topRight = copy.copy(self.predict_topRight)
        self.bottomLeft = copy.copy(self.predict_bottomLeft)
        self.bottomRight = copy.copy(self.predict_bottomRight)

    def update_predict_bbx(self, px, py):
        '''
            Description:
                update the bounding box prediction value of moving object by the new topleft
            Params:
                px(int): axis x of top vertex
                py(int): axis y of top vertex
        '''
        dx = 0
        dy = 0
        if px < 0:
            dx = -px
            px = 0
        if py < 0:
            dy = -py
            py = 0
        self.predict_bbx = BoundingBox(int(px), int(py), int(max(0,self.bounding_box.width - dx)), int(max(0,self.bounding_box.height - dy)))
        self.predict_image = self.img_full[
                            self.predict_bbx.pY:self.predict_bbx.pY + self.predict_bbx.height,
                            self.predict_bbx.pX:self.predict_bbx.pX + self.predict_bbx.width
                        ]
        [x,y] = [self.predict_bbx.pX, self.predict_bbx.pY]
        self.predict_center = np.round(np.array([[x + int(self.predict_bbx.width / 2)], [y + int(self.predict_bbx.height / 2)]]))
        self.predict_topLeft = np.round(np.array([[x], [y]]))
        self.predict_topRight = np.round(np.array([[x + self.predict_bbx.width], [y]]))
        self.predict_bottomLeft = np.round(np.array([[x], [y + self.predict_bbx.height]]))
        self.predict_bottomRight = np.round(np.array([[x + self.predict_bbx.width], [y + self.predict_bbx.height]]))

    def compare_other(self, other_moving_obj):
        '''
            Description:
                compare this moving object with another moving object
            Params:
                other_moiving_obj(MovingObject): the other moving object that need to compute the different
            Returns:
                return the different between two moving object

        '''
        feature_matching = FeatureMatching()
        diff = maxsize
        if AUTO_FOV_COMPUTE is True:
            # in the case fov is compute automatically
            if self.vector is not None and other_moving_obj.vector is not None:
                diff = pow(self.vector[0] - other_moving_obj.vector[0], 2) + pow(self.vector[1] - other_moving_obj.vector[1], 2)
            diff += feature_matching.compare_object(self, other_moving_obj, LIST_FEATURE_EXTRACTION[3])
        else:
            h,w,_ = self.img_full.shape
            if self.exist_in == 1:
                if self.bounding_box.pX < (w - self.bounding_box.pXmax):
                    # object exist from the left side of camera 1 (edge BC)
                    diff = other_moving_obj.list_distance[1]
                else:
                    # object exist from the right side of camera 1 (edge DA)
                    diff = other_moving_obj.list_distance[3]
            elif self.exist_in == 2:
                if self.bounding_box.pX < (w - self.bounding_box.pXmax):
                    # object exist from the left side of camera 1 (edge DA)
                    diff = other_moving_obj.list_distance[3]
                else:
                    # object exist from the right side of camera 1 (edge BC)
                    diff = other_moving_obj.list_distance[1]
        return diff

    def compare_features(self, other_moving_obj):
        '''
            Description:
                compare this moving object with another moving object
            Params:
                other_moiving_obj: the other moving object that need to compute the different
            Returns:
                return the different between two moving object

        '''
        result = 0;
        if AUTO_FOV_COMPUTE is True:
            feature_matching = FeatureMatching()
            HU_diff = feature_matching.compare_object(self, other_moving_obj, LIST_FEATURE_EXTRACTION[1])
            HU_diff = np.sum(HU_diff)
            CH_diff = feature_matching.compare_object(self, other_moving_obj, LIST_FEATURE_EXTRACTION[2])
            SI_diff = feature_matching.compare_object(self, other_moving_obj, LIST_FEATURE_EXTRACTION[3])
            result = HU_diff * WEIGHTS[0] + WEIGHTS[1]/CH_diff + SI_diff * WEIGHTS[2]
        return result

    def distance_to_fov_edge(self, fov):
        '''
            Description:
                compute the distances from this object (bottom-left) to each edge of the fov
            Params:
                fov(FOV): the fov object that is refered to compute
        '''
        if AUTO_FOV_COMPUTE is False:
            bl = (self.bounding_box.pX, self.bounding_box.pYmax)    # bottom right vertex
            distance1 = fov.AB.distance_from_point(bl)
            distance2 = fov.BC.distance_from_point(bl)
            distance3 = fov.CD.distance_from_point(bl)
            distance4 = fov.DA.distance_from_point(bl)
            self.list_distance = [distance1, distance2, distance3, distance4]

