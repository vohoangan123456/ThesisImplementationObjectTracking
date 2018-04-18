from shapely.geometry import Point
import numpy as np
LIST_FEATURE_EXTRACTION = ['momentColor', 'huInvariance', 'colorHistogram', 'sift', 'surf']
WEIGHTS = [1, 2, 2]
class FeatureExtractor:
    def __init__(self, image):
        self.image = image
        self.gray_image = utils.to_gray(self.image)
        return

    def extract_moment_color(self):
        mean, variance = cv2.meanStdDev(self.image)
        skewness = variance ** (2 / 3)
        momentColor = [
            mean, variance, skewness
        ]
        return momentColor

    def extract_hu_moment(self):
        return cv2.HuMoments(cv2.moments(self.gray_image)).flatten()

    def extract_color_histogram(self):
        hist = cv2.calcHist([self.image], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def extract_sift_features(self):
        sift = cv2.xfeatures2d.SIFT_create()
        kp, desc = sift.detectAndCompute(self.gray_image, None)
        return kp, desc #keypoints and descriptors

    def extract_surf_features(self):
        surf = cv2.xfeatures2d.SURF_create()
        kp, desc = surf.detectAndCompute(self.gray_image, None)
        return kp, desc #keypoints and descriptors

class FeatureMatching:
    def __init__(self):
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def compute_diff(self, desc1, desc2):
        if desc1 == None or desc2 == None:
            return 0
        matches = self.bf_matcher.match(desc1, desc2)
        #return_value = sum(c.distance for c in matches) / (len(matches) + 0.00000000000001)
        return len(matches)

    # obj1: MovingObject
    # obj2: MovingObject
    def compare_object(self, obj1: MovingObject, obj2: MovingObject, feature_label:str):
        diff_value = 0
        if feature_label == LIST_FEATURE_EXTRACTION[1]: #hu invariant
            diff_value = abs(np.array(obj1.hu_moment_feature) - np.array(obj2.hu_moment_feature))
        elif feature_label == LIST_FEATURE_EXTRACTION[2]:   #color histogram
            diff_value = cv2.compareHist(obj1.color_histogram_feature, obj2.color_histogram_feature, cv2.HISTCMP_INTERSECT)
        elif feature_label == LIST_FEATURE_EXTRACTION[3]:   #sift
            diff_value = self.compute_diff(obj1.surf_feature['Descriptor'], obj2.surf_feature['Descriptor'])
        return diff_value

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
        self.HU_feature = None  # hu invariants
        self.CH_feature = None  # color histogram
        self.SI_feature = None  # sift
        self.vector = None      # vector from moving object to the nearest point in FOV

    def set_label(self, label:str):
        self.label = label

    def get_feature(self):
        feature_extractor = FeatureExtractor(self.image)
        self.CH_feature = feature_extractor.extract_color_histogram()
        self.HU_feature = feature_extractor.extract_hu_moment()
        self.SI_feature = feature_extractor.extract_sift_features()

    def compare_other(self, other_moving_obj):
        feature_matching = FeatureMatching()
        HU_diff = feature_matching.compare_object(self, other_moving_obj, LIST_FEATURE_EXTRACTION[1])
        CH_diff = feature_matching.compare_object(self, other_moving_obj, LIST_FEATURE_EXTRACTION[2])
        SI_diff = feature_matching.compare_object(self, other_moving_obj, LIST_FEATURE_EXTRACTION[3])
        return HU_diff * WEIGHTS[0] + CH_diff * WEIGHTS[1] + SI_diff * WEIGHTS[2]

