from shapely.geometry import Point
import numpy as np
from sys import maxsize
from math import pow
import cv2
LIST_FEATURE_EXTRACTION = ['momentColor', 'huInvariance', 'colorHistogram', 'sift', 'surf']
WEIGHTS = [100, 2, 2, 3] # weight of the features that use for compute different between two object
THRESHOLD_Y = 20
class FeatureExtractor:
    def __init__(self, image):
        self.image = image
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_HSV2RGB)
        return

    def extract_hu_moment(self):
        '''
            Description:
                extract hu invariant moments
            Return:
                Hu sevent invariant moment vector
        '''
        return cv2.HuMoments(cv2.moments(self.gray_image)).flatten()

    def extract_color_histogram(self):
        '''
            Description:
                extract color histogram
            Return:
                color histogram
        '''
        hist = cv2.calcHist([self.hsv_image], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def extract_sift_features(self):
        '''
            Description:
                extract sift
            Return:
                sift keypoint and sift descriptor
        '''
        sift = cv2.xfeatures2d.SIFT_create()
        kp, desc = sift.detectAndCompute(self.gray_image, None)
        return kp, desc

class FeatureMatching:
    def __init__(self):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.bf_matcher = cv2.BFMatcher()

    def compute_diff(self, desc1, desc2):
        '''
            Description:
                compute the different of two sift descriptor
            Params:
                desc1: description 1
                desc2: description 2
            Returns:
                the number of matching points
        '''
        if desc1 is None or desc2 is None:
            return 0
        
        #matches = self.flann.knnMatch(desc1, desc2, k=2)
        matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
        good = []
        if len(matches) > 0 and len(matches[0]) == 2:
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
        
        return 50 - len(good)

    def compare_object(self, obj1, obj2, feature_label:str):
        '''
            Description:
                compare two object with the specific feature
            Params:
                obj1: moving object 1
                obj2: moving object 2
                feature_label: the name of the feature that need to be extracted
            Returns:
                the different between two moving object based on the given feature label
        '''
        diff_value = 0
        if feature_label == LIST_FEATURE_EXTRACTION[1]:     # hu invariant
            diff_value = abs(np.array(obj1.HU_feature) - np.array(obj2.HU_feature))
        elif feature_label == LIST_FEATURE_EXTRACTION[2]:   # color histogram
            diff_value = cv2.compareHist(obj1.CH_feature, obj2.CH_feature, cv2.HISTCMP_INTERSECT)
        elif feature_label == LIST_FEATURE_EXTRACTION[3]:   # sift
            diff_value = self.compute_diff(obj1.SI_feature[1], obj2.SI_feature[1])
        return diff_value

class BoundingBox:
    def __init__(self, pX, pY, width, height):
        self.pX = pX
        self.pY = pY
        self.pXmax = pX + width
        self.pYmax = pY + height
        self.width = width
        self.height = height
        self.center:Point = Point(pX, pY + height)
        self.area = width * height
        self.is_under_of_occlusion = False  # flag check this bbx is under of another bbx
        self.is_topleft_occlusion = None    # flag check this bbx is under of another bbx and position is in topleft

    def check_intersec_each_other(self, another_bbx):
        '''
            Description:
                check the whether this bbx intersect with another_bbx
            Params:
                another_bbx: the another bbx that intersect with this bbx
            Returns:
                bool: two bbx is intersected or not
        '''
        dx = min(self.pXmax, another_bbx.pXmax) - max(self.pX, another_bbx.pX)  # = delta x = min of top left x - max of bottom right x
        dy = min(self.pYmax, another_bbx.pYmax) - max(self.pY, another_bbx.pY)  # = delta y = min of top left y - max of bottom right y
        if (dx>=0) and (dy>=0):
            return True
        return False
    def check_direction_of_intersec(self, another_bbx):
        '''
            Description:
                check the position top, left, right, bottom of this bbx to another_bbx
            params:
                another_bbx: the another bbx that intersect with this bbx
            returns:
                topLeft:bool    if True this rect is in top-left of another_bbx ortherwise is in bottom-right
        '''
        topLeft = min(self.pX, another_bbx.pX) == self.pX or min(self.pY, another_bbx.pY) == self.pY
        self.is_topleft_occlusion = topLeft
        return topLeft
    def check_behind_of_otherbbx(self, another_bbx):
        '''
            Description:
                check the whether this bbx is behind or in front of another_bbx
            Params:
                another_bbx: the another bbx that intersect with this bbx
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
        self.is_under_of_occlusion = result
        return result;

class MovingObject:
    def __init__(self, image, bounding_box:BoundingBox):
        self.bounding_box = bounding_box
        self.image = image[
                            self.bounding_box.pY:self.bounding_box.pY + self.bounding_box.height,
                            self.bounding_box.pX:self.bounding_box.pX + self.bounding_box.width
                        ]
        self.label = 0
        #[x,y] = [bounding_box.center.x, bounding_box.center.y]
        #self.center = np.round(np.array([[x], [y]]))
        [x,y] = [bounding_box.pX, bounding_box.pY]
        self.center = np.round(np.array([[x], [y + bounding_box.height]]))
        self.topLeft = np.round(np.array([[x], [y]]))
        self.topRight = np.round(np.array([[x + bounding_box.width], [y]]))
        self.bottomLeft = np.round(np.array([[x], [y + bounding_box.height]]))
        self.bottomRight = np.round(np.array([[x + bounding_box.width], [y + bounding_box.height]]))

        self.HU_feature = None  # hu invariants
        self.CH_feature = None  # color histogram
        self.SI_feature = None  # sift
        self.vector = None      # vector from moving object to the nearest point in FOV
        self.is_in_fov = False
        self.confidence = 0

    def set_label(self, label:str):
        '''
            Description:
                set label for moving object
            Params:
                label: label that need to set for object
        '''
        self.label = label
    def set_confidence(self, confidence):
        self.confidence = confidence

    def set_vector(self, nearest_point):
        '''
            Description:
                set the vector from the moving object to the nearest point in the boundary of fov
            Params:
                nearest_point: the nearest point to moving object
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

    def compare_other(self, other_moving_obj):
        '''
            Description:
                compare this moving object with another moving object
            Params:
                other_moiving_obj: the other moving object that need to compute the different
            Returns:
                return the different between two moving object

        '''
        feature_matching = FeatureMatching()
        HU_diff = feature_matching.compare_object(self, other_moving_obj, LIST_FEATURE_EXTRACTION[1])
        HU_diff = np.sum(HU_diff)
        CH_diff = feature_matching.compare_object(self, other_moving_obj, LIST_FEATURE_EXTRACTION[2])
        SI_diff = feature_matching.compare_object(self, other_moving_obj, LIST_FEATURE_EXTRACTION[3])
        vector_diff = maxsize
        if self.vector is not None and other_moving_obj.vector is not None:
            vector_diff = pow(self.vector[0] - other_moving_obj.vector[0], 2) + pow(self.vector[1] - other_moving_obj.vector[1], 2)
        return HU_diff * WEIGHTS[0] + CH_diff * WEIGHTS[1] + SI_diff * WEIGHTS[2] + vector_diff * WEIGHTS[3]

    def compare_other_without_vector(self, other_moving_obj):
        '''
            Description:
                compare this moving object with another moving object
            Params:
                other_moiving_obj: the other moving object that need to compute the different
            Returns:
                return the different between two moving object

        '''
        feature_matching = FeatureMatching()
        #HU_diff = feature_matching.compare_object(self, other_moving_obj, LIST_FEATURE_EXTRACTION[1])
        #HU_diff = np.sum(HU_diff)
        CH_diff = feature_matching.compare_object(self, other_moving_obj, LIST_FEATURE_EXTRACTION[2])
        #SI_diff = feature_matching.compare_object(self, other_moving_obj, LIST_FEATURE_EXTRACTION[3])
        return CH_diff * WEIGHTS[1]# + SI_diff * WEIGHTS[2]
    def compare_other_in_one_camera(self, other_moving_obj):
        '''
            Description:
                compare this moving object with another moving object
            Params:
                other_moiving_obj: the other moving object that need to compute the different
            Returns:
                return the different between two moving object

        '''
        feature_matching = FeatureMatching()
        HU_diff = feature_matching.compare_object(self, other_moving_obj, LIST_FEATURE_EXTRACTION[1])
        HU_diff = np.sum(HU_diff)
        CH_diff = feature_matching.compare_object(self, other_moving_obj, LIST_FEATURE_EXTRACTION[2])
        SI_diff = feature_matching.compare_object(self, other_moving_obj, LIST_FEATURE_EXTRACTION[3])
        return CH_diff * WEIGHTS[1] + SI_diff * WEIGHTS[2]

