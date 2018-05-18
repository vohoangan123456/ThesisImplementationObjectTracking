import cv2
import numpy as np
from my_working_space.kalman_filter.moving_object import MovingObject, BoundingBox
from my_working_space.kalman_filter.field_of_view import CommonFOV
from random import randint
THRESHOLD_CONFIDENCE = 0.5
THRESHOLD_SIZE = 4000
class yolo_detector:
    def __init__(self, tfNet):
        self.tfNet = tfNet
        self.list_moving_obj = []       # list all moving object detected
        self.frame_index = 0            # frame index of video
        self.detections = []            # list detections value
        self.list_under_occlusion = []  # list object is under occlusion
        self.list_not_occlusion = []    # list object not need to concern about the occlusion

    def detect(self, frame):
        '''
            Description:
                Detect the moving object in the given frame
            Params:
                frame: the given frame
        '''
        self.frame_index += 1
        if self.frame_index == 34:
            x = 1
        cv2.putText(frame, str('frame: {0}'.format(self.frame_index)), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        print('--------------------------------', str(self.frame_index), '-----------------------------')
        print('# detection')
        results = self.tfNet.return_predict(frame)
        self.list_moving_obj = []
        for result in results:
            label = result['label']
            confidence = result['confidence']
            if label == 'person':
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])                
                if confidence >= 0.59:
                    bounding_box = BoundingBox(tl[0], tl[1], abs(tl[0] - br[0]), abs(tl[1] - br[1]))
                    moving_obj = MovingObject(frame, bounding_box)
                    moving_obj.set_confidence(confidence)
                    moving_obj.get_feature()
                    self.list_moving_obj.append(moving_obj);
                print(str("%.2f" % confidence), str('({0},{1})'.format(tl[0], tl[1])))
        self.find_object_under_occlusion()
        if len(self.list_moving_obj) == 2 and len(self.list_under_occlusion) != 0:
            x = 12
        # Slower the FPS
        cv2.waitKey(50)

    def convert_to_detection(self):
        '''
            Description:
                convert list moving object to list detection value
        '''
        self.detections = []
        for obj in self.list_moving_obj:
            x1 = obj.bounding_box.pX
            y1 = obj.bounding_box.pY
            x2 = x1 + obj.bounding_box.width
            y2 = y1 + obj.bounding_box.height
            score = obj.confidence
            self.detections.append([x1, y1, x2, y2, score])
        self.detections = np.array(self.detections)

    def find_object_under_occlusion(self):
        '''
            Description:
                find the object is under occlusion that need to be concern when tracking
        '''
        self.list_under_occlusion = []
        for i, obj1 in enumerate(self.list_moving_obj):
            for j, obj2 in enumerate(self.list_moving_obj):
                if i != j and obj1.bounding_box.check_intersec_each_other(obj2.bounding_box) and obj1.bounding_box.check_behind_of_otherbbx(obj2.bounding_box) and obj1.bounding_box.is_under_of_occlusion is False:
                    # check if the obj1 is intersected with obj2 and it's under the obj2
                    obj1.bounding_box.is_under_of_occlusion = True
                    obj1.bounding_box.check_direction_of_intersec(obj2.bounding_box)
                    obj1.bounding_box.get_overlap_area(obj2.bounding_box)
                    self.list_under_occlusion.append(obj1)
        self.list_not_occlusion = set(self.list_moving_obj) - set(self.list_under_occlusion)

class detection_write:
    def __init__(self, tfNet, camId, rd_number):
        self.tfNet = tfNet
        self.list_moving_obj = []       # list all moving object detected
        self.frame_index = 0            # frame index of video
        self.detections = []            # list detections value
        self.list_under_occlusion = []  # list object is under occlusion
        self.list_not_occlusion = []    # list object not need to concern about the occlusion
        self.file = open("./videos/detection_code{0}_cam{1}.txt".format(rd_number, camId), "w")

    def release_file(self):
        self.file.close()

    def detect(self, frame):
        '''
            Description:
                Detect the moving object in the given frame
            Params:
                frame: the given frame
        '''
        self.frame_index += 1
        results = self.tfNet.return_predict(frame)
        list_detector= []
        for result in results:
            label = result['label']
            confidence = result['confidence']
            if label == 'person':
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])                
                if confidence >= 0.5:
                    line = "{0}, {1}, {2}, {3}, {4}, {5}\n".format(self.frame_index, confidence, tl[0], tl[1], br[0], br[1])
                    list_detector.append(line)
        self.file.writelines(list_detector)
        # Slower the FPS
        cv2.waitKey(18)

class detection_read:
    def __init__(self, file_txt):
        self.file = open(file_txt, "r")
        self.list_detection = []
        self.extract_detection()
        self.list_moving_obj = []       # list all moving object detected
        self.frame_index = 0            # frame index of video
        self.detections = []            # list detections value
        self.list_under_occlusion = []  # list object is under occlusion
        self.list_not_occlusion = []    # list object not need to concern about the occlusion

    def extract_detection(self):
        for lines in self.file:
            lines = lines[:-1].split(', ')
            frame_index = int(lines[0])
            confidence = float(lines[1])
            tl = (int(lines[2]), int(lines[3]))
            br = (int(lines[4]), int(lines[5]))
            result = {
                "frame": frame_index,
                "confidence": confidence,
                "topleft": tl,
                "bottomright": br
                }
            self.list_detection.append(result)
    def detect(self, frame):
        '''
            Description:
                Detect the moving object in the given frame
            Params:
                frame: the given frame
        '''
        self.frame_index += 1
        #if self.frame_index <= 3680:
        #    return
        cv2.putText(frame, str('frame: {0}'.format(self.frame_index)), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        print('--------------------------------', str(self.frame_index), '-----------------------------')
        print('# detection')
        results = [item for item in self.list_detection if item["frame"] == self.frame_index]
        self.list_moving_obj = []
        for result in results:
            confidence = result['confidence']
            tl = result['topleft']
            br = result['bottomright']
            bounding_box = BoundingBox(tl[0], tl[1], abs(tl[0] - br[0]), abs(tl[1] - br[1]))
            if confidence >= THRESHOLD_CONFIDENCE and bounding_box.area > THRESHOLD_SIZE:
                moving_obj = MovingObject(frame, bounding_box)
                moving_obj.set_confidence(confidence)
                moving_obj.get_feature()
                self.list_moving_obj.append(moving_obj);
            print(str("%.2f" % confidence), str('({0},{1})'.format(tl[0], tl[1])))
        self.find_object_under_occlusion()
        # Slower the FPS
        cv2.waitKey(1)

    def convert_to_detection(self):
        '''
            Description:
                convert list moving object to list detection value
        '''
        self.detections = []
        for obj in self.list_moving_obj:
            x1 = obj.bounding_box.pX
            y1 = obj.bounding_box.pY
            x2 = x1 + obj.bounding_box.width
            y2 = y1 + obj.bounding_box.height
            score = obj.confidence
            self.detections.append([x1, y1, x2, y2, score])
        self.detections = np.array(self.detections)

    def find_object_under_occlusion(self):
        '''
            Description:
                find the object is under occlusion that need to be concern when tracking
        '''
        self.list_under_occlusion = []
        for i, obj1 in enumerate(self.list_moving_obj):
            for j, obj2 in enumerate(self.list_moving_obj):
                if i != j and obj1.bounding_box.check_intersec_each_other(obj2.bounding_box) and obj1.bounding_box.check_behind_of_otherbbx(obj2.bounding_box) and obj1.bounding_box.is_under_of_occlusion is False:
                    # check if the obj1 is intersected with obj2 and it's under the obj2
                    obj1.bounding_box.is_under_of_occlusion = True
                    obj1.bounding_box.check_direction_of_intersec(obj2.bounding_box)
                    obj1.bounding_box.get_overlap_area(obj2.bounding_box)
                    self.list_under_occlusion.append(obj1)
        self.list_not_occlusion = set(self.list_moving_obj) - set(self.list_under_occlusion)
