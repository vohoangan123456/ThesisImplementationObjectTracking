import cv2
import my_working_space.My_Utils as utils
from darkflow.net.build import TFNet
import numpy as np
import time
import sys
import imutils
import operator
from collections import Counter

MAX_NUMBER_OBJECT_IN_CLUSTER = 5
LIST_FEATURE_EXTRACTION = ['momentColor', 'huInvariance', 'colorHistogram', 'sift', 'surf']

class BackgroundSubtractor:
    def __init__(self, backGroundModel):
        self.firstModel = backGroundModel

    def myBGSubtractor(self, frame):
        compareFirst = cv2.absdiff(self.firstModel, frame)
        compareFirst = cv2.cvtColor(compareFirst, cv2.COLOR_BGR2GRAY)
        grayScale = cv2.inRange(compareFirst, 60, 255)
        grayScale = cv2.erode(grayScale, None, iterations=2)
        grayScale = cv2.dilate(grayScale, None, iterations=2)
        grayScale = cv2.medianBlur(grayScale, 5)
        grayScale = cv2.GaussianBlur(grayScale, (11, 11), 0)
        compareFirst = grayScale
        return compareFirst

class BoundingBox:
    def __init__(self, pX, pY, width, height):
        self.pX = pX
        self.pY = pY
        self.width = width
        self.height = height

class MovingObject:
    def __init__(self):
        self.bounding_box = None
        self.image = None
        self.crop_object_img = None
        self.color_histogram_feature = None
        self.moment_color_feature = None
        self.hu_moment_feature = None
        self.sift_feature = None
        self.surf_feature = None
        self.feature_extractor = None
        self.label = 0

    def create_object_with_boundingbox(self, image, bounding_box:BoundingBox):
        self.bounding_box = bounding_box
        self.image = image
        self.crop_object_img = image[
                               self.bounding_box.pY:self.bounding_box.pY + self.bounding_box.height,
                               self.bounding_box.pX:self.bounding_box.pX + self.bounding_box.width
                               ]
        self.feature_extractor = FeatureExtractor(self.crop_object_img)
        self.label = 0
        self.feature_extraction()

    def create_object_with_feature(self, moment_color, hu_moment, sift, surf, label):
        self.moment_color_feature = moment_color
        self.hu_moment_feature = hu_moment
        self.sift_feature = sift
        self.surf_feature = surf
        self.label = label

    def feature_extraction(self):
        self.moment_color_feature = self.feature_extractor.extract_moment_color()
        self.hu_moment_feature = self.feature_extractor.extract_hu_moment()
        self.color_histogram_feature = self.feature_extractor.extract_color_histogram()
        kp, desc = self.feature_extractor.extract_sift_features()
        self.sift_feature = {
            'KeyPoint' : kp,
            'Descriptor' : desc
        }
        kp, desc = self.feature_extractor.extract_surf_features()
        self.surf_feature = {
            'KeyPoint' : kp,
            'Descriptor' : desc
        }

    def set_label(self, label:str):
        self.label = label

class Cluster:
    def __init__(self, leader:MovingObject, label:int):
        self.leader: MovingObject = leader
        self.list_object: list[MovingObject] = [leader]
        self.max_element_no: int = MAX_NUMBER_OBJECT_IN_CLUSTER
        self.feater_matching: FeatureMatching = FeatureMatching()
        self.label: int = label

    def set_label(self, label:int):
        self.label = label

    def set_max_element_no(self, max_number_object:int):
        self.max_element_no = max_number_object

    def add_new_element(self, new_object:MovingObject):
        self.list_object.append(new_object)
        self.find_new_leader_2()

    #get the distance from element to all other element
    def get_total_distance(self, element:MovingObject, feature_label):
        total_distance:int = 0
        for obj in self.list_object:
            total_distance += self.feater_matching.compare_object(element, obj, feature_label)
        return total_distance

    def find_new_leader_2(self):
        if self.max_element_no != -1:
            self.list_object = self.list_object[-self.max_element_no:]
        # dict_temp = {}  # dictionary store element and the total distance from it to all the other element
        # for index, element in enumerate(self.list_object):
        #     dict_temp[element] = self.get_total_distance(element)
        # sorted_dict = sorted(dict_temp.items(), key=operator.itemgetter(1))
        # self.leader = sorted_dict.pop()[0]
        self.leader = self.list_object[-1]

    def find_new_leader_1(self):
        dict_temp = {}   #dictionary store element and the total distance from it to all the other element
        for index, element in enumerate(self.list_object):
            dict_temp[element] = self.get_total_distance(element)
        sorted_dict = sorted(dict_temp.items(), key=operator.itemgetter(1))
        if self.max_element_no != -1:
            dict_temp = dict((x, y) for x, y in sorted_dict)
            for key, val in dict_temp.items():
                self.list_object.append(key)
        self.leader = sorted_dict.pop()[0]

    def find_new_leader(self):
        moment_color_average = None
        hu_moment_average = None
        sift_average = None
        surf_average = None
        for index, element in enumerate(self.list_object):
            if index == 0:
                moment_color_average = element.moment_color_feature
                hu_moment_average = element.hu_moment_feature
                sift_average = element.sift_feature
                surf_average = element.surf_feature
            else:
                moment_color_average = [x + y for x, y in zip(moment_color_average, element.moment_color_feature)]
                hu_moment_average = [x + y for x, y in zip(hu_moment_average, element.hu_moment_feature)]
                # sift_average = utils.merge_two_dict(sift_average, element.sift_feature)
                # surf_average = utils.merge_two_dict(surf_average, element.surf_feature)
        length_list:int = len(self.list_object)
        moment_color_average = [x / length_list for x in moment_color_average]
        hu_moment_average = [x / length_list for x in hu_moment_average]
        self.leader = MovingObject()
        self.leader.create_object_with_feature(moment_color_average, hu_moment_average, sift_average, surf_average, self.label)

class FeatureMatching:
    def __init__(self):
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def compute_diff(self, desc1, desc2):
        if desc1 == None or desc2 == None:
            return 0
        matches = self.bf_matcher.match(desc1, desc2)
        # matches = sorted(matches, key=lambda x: x.distance)
        # matches = matches[:self.N_MATCHES]
        return_value = sum(c.distance for c in matches) / (len(matches) + 0.00000000000001)
        return return_value

    # obj1: MovingObject
    # obj2: MovingObject
    def compare_object(self, obj1: MovingObject, obj2: MovingObject, feature_label:str):
        diff_value = 0
        if feature_label == LIST_FEATURE_EXTRACTION[0]: #moment color
            diff_value = abs(np.array(obj1.moment_color_feature) - np.array(obj2.moment_color_feature))
        elif feature_label == LIST_FEATURE_EXTRACTION[1]: #hu invariant
            diff_value = abs(np.array(obj1.hu_moment_feature) - np.array(obj2.hu_moment_feature))
        elif feature_label == LIST_FEATURE_EXTRACTION[2]:   #color histogram
            diff_value = cv2.compareHist(obj1.color_histogram_feature, obj2.color_histogram_feature,
                                                   cv2.HISTCMP_INTERSECT)
        elif feature_label == LIST_FEATURE_EXTRACTION[3]:   #sift
            diff_value = self.compute_diff(obj1.surf_feature['Descriptor'], obj2.surf_feature['Descriptor'])
        elif feature_label == LIST_FEATURE_EXTRACTION[4]:   #surf
            diff_value = self.compute_diff(obj1.sift_feature['Descriptor'], obj2.sift_feature['Descriptor'])
        # total_diff = np.sum(diff_hu_moment) + np.sum(diff_moment_color) + diff_sift + diff_surf
        # diff_value = np.sum(diff_moment_color) + (np.sum(diff_hu_moment) * 10**6) + (diff_surf * 10) + (diff_sift / 200)
        return diff_value

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

class KNNAlgorithm:
    # list_trained_model : list of the cluster
    # testing_model : the model need to be assigned
    # k : the k-nearest value
    def __init__(self, list_trained_model:list, testing_model:MovingObject, k:int = 1):
        self.list_model:list[Cluster] = list_trained_model
        self.test_model:MovingObject = testing_model
        self.k:int = k
        self.feature_matching:FeatureMatching = FeatureMatching()
        self.list_knn = {}

    def set_k(self, k:int):
        self.k = k

    def run(self):
        self.list_knn = {}
        for item in LIST_FEATURE_EXTRACTION:
            dict = {}
            for cluster in self.list_model:
                dict[cluster.label] = cluster.get_total_distance(self.test_model, item)
            sorted_dict = sorted(dict.items(), key=operator.itemgetter(1))
            k_nearest_list = sorted_dict[:self.k]
            self.list_knn[item] = k_nearest_list
        return self.list_knn
    def vote(self):
        for label in self.list_knn:
            for item in label:
                print(item[0], item[1])

class LeaderAlgorithm:
    def __init__(self, threshold:int):
        self.list_class:list[Cluster] = []   #list store all class, element is Cluster
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)   #find the similar using surf and sift
        self.feature_matching:FeatureMatching = FeatureMatching()
        self.current_label:int = 1
        self.threshold:int = threshold

    # moving_object : MovingObject
    def assign_label(self, moving_object:MovingObject):
        if len(self.list_class) == 0:
            moving_object.set_label(self.current_label)
            object_class = Cluster(moving_object, self.current_label)
            self.list_class.append(object_class)
            self.current_label += 1
            return
        print('-------------------------------------start-----------------------------------')
        min_diff_value = sys.maxsize
        min_diff_index = -1
        knn_algorithm:KNNAlgorithm = KNNAlgorithm(self.list_class, moving_object)
        #return k-nearest neighbour
        list_nearest = knn_algorithm.run()
        # nearest = list_nearest.pop()
        nearest = list_nearest[-1]
        if nearest[1] >= self.threshold:
            index_of_nearest = -1
            for index in range(0, len(self.list_class)):
                if self.list_class[index].label == nearest[0]:
                    index_of_nearest = index
                    break
            moving_object.set_label(nearest[0])
            self.list_class[index_of_nearest].add_new_element(moving_object)
        else:
            moving_object.set_label(self.current_label)
            object_class = Cluster(moving_object, self.current_label)
            self.list_class.append(object_class)
            self.current_label += 1

        print('label', nearest[1], nearest[0])
        print('-----------------------------------------end------------------------------------------')

class ObjectDetector:
    def __init__(self, video_path:str, options, threshold = 10):
        self.video_path = video_path
        self.options = options
        # self.tfNet = TFNet(self.options)
        self.leader_algorithm = LeaderAlgorithm(threshold)

    def detect_object(self):
        capture = cv2.VideoCapture(self.video_path)
        colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
        bgSubtractor = None
        while (capture.isOpened()):
            stime = time.time()
            ret, frame = capture.read()
            #-- new approach
            frame = imutils.resize(frame, width=600)  # scale the frame to a fixed pixel
            if bgSubtractor == None:
                bgSubtractor = BackgroundSubtractor(frame)
            foreGround = bgSubtractor.myBGSubtractor(frame)
            # find contours in the backgroundsubtractFrame and initialize the current
            cnts = cv2.findContours(foreGround.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            # find bounding box
            # loop all contours to draw rectangle for moving object
            for c in cnts:
                (pX, pY, width, height) = cv2.boundingRect(c)
                # only proceed if the radius meets a minimum size
                if width * height > 3050:

                    # process_assign_label_here
                    bounding_box = BoundingBox(pX, pY, width, height)
                    moving_object = MovingObject()
                    moving_object.create_object_with_boundingbox(frame, bounding_box)
                    # cv2.imshow('moving_obj', moving_object.image)
                    cv2.imshow('moving_obj_crop', moving_object.crop_object_img)
                    self.leader_algorithm.assign_label(moving_object)

                    #show detector
                    cv2.rectangle(frame, (pX, pY), (pX + width, pY + height), (0, 255, 0), 2)
                    cv2.putText(frame, str(moving_object.label), (pX, pY), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255),
                                1)

            # show the frame to our screen and increment the frame counter
            # cv2.imshow("Background subtraction's video", foreGround)
            cv2.imshow("Result Video", frame)

            key = cv2.waitKey(18) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                break

            # if ret:
            #     frame = cv2.resize(frame, (600, 600))
            #     results = self.tfNet.return_predict(frame)
            #     for color, result in zip(colors, results):
            #         label = result['label']
            #         if label == 'person':
            #             tl = (result['topleft']['x'], result['topleft']['y'])
            #             br = (result['bottomright']['x'], result['bottomright']['y'])
            #             bounding_box = BoundingBox(tl[0], tl[1], abs(tl[0] - br[0]), abs(tl[1] - br[1]))
            #             # process_assign_label_here
            #             # bounding_box = BoundingBox(pX, pY, width, height)
            #             moving_object = MovingObject()
            #             moving_object.create_object_with_boundingbox(frame, bounding_box)
            #             cv2.imshow('moving_obj_crop', moving_object.crop_object_img)
            #             self.leader_algorithm.assign_label(moving_object)
            #             frame = cv2.rectangle(frame, tl, br, color, 2)
            #             frame = cv2.putText(frame, (label + '-' + str(moving_object.label)), tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            #     cv2.imshow('frame', frame)
            #     print('FPS {:.1f}'.format(1 / (time.time() - stime + 0.0000000000001)))
            #     if cv2.waitKey(18) & 0xFF == ord('q'):
            #         break
            # else:
            #     capture.release()
            #     cv2.destroyAllWindows()
            #     break
