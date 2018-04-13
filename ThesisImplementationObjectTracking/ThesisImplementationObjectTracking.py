
import cv2
from my_working_space.Main_Functions import ObjectDetector, FeatureExtractor
import time

options = {
    'model': 'cfg/tiny-yolo-voc.cfg',
    'load': 'bin/tiny-yolo-voc.weights',
    'threshold': 0.15,
    'gpu': 1.0
}
# options = {
#     'model': 'cfg/yolo.cfg',
#     'load': 'bin/yolo.weights',
#     'threshold': 0.15,
#     'gpu': 1.0
# }
# threshold contain 3 main element
# color-moment: 2
# hu-moment: 1.0e-05
# surf: 0.02

threshold = 4
video_path = 'videofile.mp4'
object_detector = ObjectDetector(video_path, options, threshold)
start_time = time.time()
print(start_time, '-----------------------------------------')
object_detector.detect_object()
print(str(time.time() - start_time), '-----------', time.time())