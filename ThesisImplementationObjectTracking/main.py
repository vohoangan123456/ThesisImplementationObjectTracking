import my_working_space.kalman_filter.object_tracking as obj_tracker

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.20,
    'gpu': 1.0
}
#options = {
#    'model': 'cfg/tiny-yolo-voc.cfg',
#    'load': 'bin/tiny-yolo-voc.weights',
#    'threshold': 0.20,
#    'gpu': 1.0
#}

threshold = 20
video_path1 = './videos/sample_video/campusc7_c0_edit.avi'
video_path2 = './videos/sample_video/campusc7_c1_edit.avi'

#obj_tracker.tracking_object(video_path1, video_path2, options)
#obj_tracker.tracking_single_camera('./videos/video2.avi', options)
obj_tracker.multi_tracking_object(video_path1, video_path2, options)
