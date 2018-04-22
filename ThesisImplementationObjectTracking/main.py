import my_working_space.kalman_filter.object_tracking as obj_tracker

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.20,
    'gpu': 1.0
}

threshold = 20
video_path1 = './videos/devide_video2_video1.avi'
video_path2 = './videos/devide_video2_video2.avi'

obj_tracker.tracking_object(video_path1, video_path2, options)
#obj_tracker.tracking_single_camera(video_path1, options)