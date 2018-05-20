'''
    File name         : features.py
    File Description  : handle the feature and feature extractor for moving object 
    Author            : An Vo
    Date created      : 19/04/2018
    Python Version    : 3.6
'''
import cv2
from my_implementations.common.global_config import *

class FeatureExtractor:
    def __init__(self, image):
        '''
            Description:
                Initialize feature extractor of moving object
            Params:
                image: the image get from camera frame
        '''
        self.image = image                                              # original image
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # gray image
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_HSV2RGB)    # hsv image
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
        hist = cv2.calcHist([self.hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
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
