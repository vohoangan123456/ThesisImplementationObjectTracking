import my_working_space.get_points as get_points
import cv2
import my_working_space.My_Utils as utils
import numpy as np
import sys
import imutils
import operator
from darkflow.net.build import TFNet
import time
KNN_PARAM = 1
THRESHOLD = 10

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
        self.center:Location = Location(pX + width / 2, pY + height / 2)

class MovingObject:
    def __init__(self):
        self.bounding_box:BoundingBox = None
        self.image = None
        self.crop_object_img = None
        self.label:str = 0
        self.location:Location = None
        self.is_on_screen:bool = True

    def create_object_with_boundingbox(self, image, bounding_box:BoundingBox):
        self.bounding_box = bounding_box
        self.image = image
        self.crop_object_img = image[
                               self.bounding_box.pY:self.bounding_box.pY + self.bounding_box.height,
                               self.bounding_box.pX:self.bounding_box.pX + self.bounding_box.width
                               ]
        self.location = bounding_box.center
    def set_label(self, label:str):
        self.label = label
    
    def set_location(self, location:Location):
        self.location = location

    def set_status(self, status:bool):
        self.is_on_screen = status

class KNNAlgorithm:
    def __init__(self, k:int = KNN_PARAM):
        self.k:int = k

    def set_k(self, k:int):
        self.k = k

    def distance(locationA, locationB):
        return numpy.sqrt((locationA.pX-locationB.pX)**2 + (locationA.pY - locationB.pY)**2)

    def find_k_nearest_obj(self, list_model:list, train_model:MovingObject):
        list_knn = {}
        center_min_distance = sys.maxsize
        for index, item in enumerate(list_model):
            dist = np.sqrt((item.location.pX-train_model.location.pX)**2 + (item.location.pY - train_model.location.pY)**2)
            if dist < center_min_distance:
                center_min_distance = dist
                list_knn[0] = item
                list_knn[1] = dist
        return list_knn

class MyAlgorithm:
    def __init__(self):
        self.current_label:int = 1
        self.list_model:list[MovingObject] = []
        self.knn_algorithm:KNNAlgorithm = KNNAlgorithm(KNN_PARAM)

    def add_new_object(self, moving_obj):
        self.list_model.append(moving_obj)

    def run_knn(self, train_model:MovingObject):
        knn_obj = self.knn_algorithm.find_k_nearest_obj(self.list_model, train_model)
        if not knn_obj or knn_obj[1] > THRESHOLD:     #new object
            train_model.set_label(str(self.current_label))
            self.list_model.append(train_model)
            self.current_label += 1
            return
        #update object of label
        train_model.set_label(knn_obj[0].label)
        for index, item in enumerate(self.list_model):
            if item.label == train_model.label:
                self.list_model[index] = train_model
                return
        

#def get_points_process(img):
#    points = get_points.run(img)
#    return points

def run(source=0, dispLoc=False):
    # Create the VideoCapture object
    cam = cv2.VideoCapture(source)
    my_process = MyAlgorithm()
    while True:
        # Read frame from device or file
        retval, frame = cam.read()
        if not retval:
            print("Cannot capture frame device | CODE TERMINATING :(")
            exit()
        if cv2.waitKey(18) == ord('e'):
            frame = imutils.resize(frame, width=800)
            points = get_points.run(frame)
            if not points:
                print("ERROR: No object to be tracked.")
                exit()
            pX = points[0][0]
            pY = points[0][1]
            width = abs(points[0][2] - pX)
            height = abs(points[0][3] - pY)
            
            bounding_box = BoundingBox(pX, pY, width, height)
            moving_object = MovingObject()
            moving_object.create_object_with_boundingbox(frame, bounding_box)
            cv2.imshow('moving_obj_crop', moving_object.crop_object_img)
            my_process.run_knn(moving_object)

            # show detector
            cv2.rectangle(frame, (pX, pY), (pX + width, pY + height), (0, 255, 0), 2)
            cv2.putText(frame, moving_object.label, (pX, pY), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", frame)
        # Continue until the user presses ESC key
        if cv2.waitKey(1) == 27:
            break

    # Relase the VideoCapture object
    cam.release()

class ObjectTracking:
    def __init__(self, video_path:str, options, threshold = 10):
        self.video_path = video_path
        self.options = options
        self.tfNet = TFNet(self.options)
        self.my_process = MyAlgorithm()

    def detect_object(self):
        capture = cv2.VideoCapture(self.video_path)
        colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
        while (capture.isOpened()):
            stime = time.time()
            retval, frame = capture.read()
            if not retval:
                print("Cannot capture frame device | CODE TERMINATING :(")
                exit()
            frame = cv2.resize(frame, (600, 600))
            results = self.tfNet.return_predict(frame)
            for color, result in zip(colors, results):
                label = result['label']
                if label == 'person':
                    tl = (result['topleft']['x'], result['topleft']['y'])
                    br = (result['bottomright']['x'], result['bottomright']['y'])

                    bounding_box = BoundingBox(tl[0], tl[1], abs(tl[0] - br[0]), abs(tl[1] - br[1]))
                    moving_object = MovingObject()
                    moving_object.create_object_with_boundingbox(frame, bounding_box)
                    cv2.imshow('moving_obj_crop', moving_object.crop_object_img)
                    self.my_process.run_knn(moving_object)

                    # show detector
                    frame = cv2.rectangle(frame, tl, br, color, 2)
                    frame = cv2.putText(frame, moving_object.label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            #if cv2.waitKey(18) == ord('e'):
            #    frame = imutils.resize(frame, width=800)
            #    points = get_points.run(frame)
            #    if not points:
            #        print("ERROR: No object to be tracked.")
            #        exit()
            #    pX = points[0][0]
            #    pY = points[0][1]
            #    width = abs(points[0][2] - pX)
            #    height = abs(points[0][3] - pY)
            
            #    bounding_box = BoundingBox(pX, pY, width, height)
            #    moving_object = MovingObject()
            #    moving_object.create_object_with_boundingbox(frame, bounding_box)
            #    cv2.imshow('moving_obj_crop', moving_object.crop_object_img)
            #    self.my_process.run_knn(moving_object)

            #    # show detector
            #    cv2.rectangle(frame, (pX, pY), (pX + width, pY + height), (0, 255, 0), 2)
            #    cv2.putText(frame, moving_object.label, (pX, pY), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #                (0, 0, 255), 1)

            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Image", frame)
            print('FPS {:.1f}'.format(1 / (time.time() - stime + 0.0000000000001)))
            # Continue until the user presses ESC key
            if cv2.waitKey(1) == 27:
                break

#if __name__ == "__main__":
#    source = './videos/videofile_inroom.avi'
#    run(source, None)

#frame = cv2.resize(frame, (600, 600))
#                results = self.tfNet.return_predict(frame)
#                for color, result in zip(colors, results):
#                    label = result['label']
#                    if label == 'person':
#                        tl = (result['topleft']['x'], result['topleft']['y'])
#                        br = (result['bottomright']['x'], result['bottomright']['y'])
#                        bounding_box = BoundingBox(tl[0], tl[1], abs(tl[0] - br[0]), abs(tl[1] - br[1]))
#                        moving_object = MovingObject()
#                        moving_object.create_object_with_boundingbox(frame, bounding_box)
#                        cv2.imshow('moving_obj_crop', moving_object.crop_object_img)
#                        my_process.run_knn(moving_object)

#                        # show detector
#                        cv2.rectangle(frame, (pX, pY), (pX + width, pY + height), (0, 255, 0), 2)
#                        cv2.putText(frame, moving_object.label, (pX, pY), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                                    (0, 0, 255), 1)