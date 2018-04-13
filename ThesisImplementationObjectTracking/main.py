#import my_working_space.select_object as algorithm
#from darkflow.net.build import TFNet
#import cv2
#import time
import my_working_space.kalman_filter.object_tracking as obj_tracker

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.15,
    'gpu': 1.0
}

threshold = 20
video_path = './videos/videofile_intown.avi'

obj_tracker.tracking_object(video_path, options)

#object_detector = algorithm.ObjectTracking(video_path, options, threshold)
#object_detector.detect_object()

#source = './videos/videofile_inroom.avi'
#algorithm.run(source, None)