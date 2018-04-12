import my_working_space.select_object as algorithm
import my_working_space.kalman_filter.object_tracking as obj_tracker
import cv2
from darkflow.net.build import TFNet
import time

options = {
    'model': 'cfg/tiny-yolo-voc.cfg',
    'load': 'bin/tiny-yolo-voc.weights',
    'threshold': 0.15,
    'gpu': 1.0
}

threshold = 10
video_path = './videos/videofile_inroom.avi'

obj_tracker.tracking_object(video_path, options)

#object_detector = algorithm.ObjectTracking(video_path, options, threshold)
#object_detector.detect_object()

#source = './videos/videofile_inroom.avi'
#algorithm.run(source, None)