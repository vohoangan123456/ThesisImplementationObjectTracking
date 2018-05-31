#import my_working_space.kalman_filter.object_tracking as obj_tracker
from my_implementations import main_process as obj_tracker
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

def read(options):
    # other camera
    #video_path1 = './videos/sample_video/edit_1_campus4-c0.avi'
    #video_path2 = './videos/sample_video/edit_1_campus4-c1.avi'
    #detection_path1 = './videos/detections/detection_edit_1_campus4-c0_code4_cam1.txt'
    #detection_path2 = './videos/detections/detection_edit_1_campus4-c1_code4_cam2.txt'

    #video_path1 = './videos/sample_video/campusc7_c0_edit.avi'
    #video_path2 = './videos/sample_video/campusc7_c1_edit.avi'
    #detection_path1 = './videos/detections/detection_campusc7_c0_edit_code40_cam1.txt'
    #detection_path2 = './videos/detections/detection_campusc7_c1_edit_code40_cam2.txt'

    #video_path1 = './videos/sample_video/campusc7_c0_edit_3300.avi'
    #video_path2 = './videos/sample_video/campusc7_c1_edit_3300.avi'
    #detection_path1 = './videos/detections/detection_campusc7_c0_edit_3300_code28_cam1.txt'
    #detection_path2 = './videos/detections/detection_campusc7_c1_edit_3300_code28_cam2.txt'
    #obj_tracker.multi_tracking_object_read(video_path1, video_path2, detection_path1, detection_path2, options)

    # khai's camera
    #video_path1 = './videos/my_video/cam1_right_video1_part_1.avi'
    #video_path2 = './videos/my_video/cam1_right_video2_part_1.avi'
    #detection_path1 = './videos/detections/detection_cam1_right_video1_part_1_cam1.txt'
    #detection_path2 = './videos/detections/detection_cam1_right_video2_part_1_cam2.txt'
    #obj_tracker.multi_tracking_object_read(video_path1, video_path2, detection_path1, detection_path2, options)

    #video_path1 = './videos/my_video/cam1_right_video1_part_2.avi'
    #video_path2 = './videos/my_video/cam1_right_video2_part_2.avi'
    #detection_path1 = './videos/detections/detection_cam1_right_video1_part_2_cam1.txt'
    #detection_path2 = './videos/detections/detection_cam1_right_video2_part_2_cam2.txt'
    #obj_tracker.multi_tracking_object_read(video_path1, video_path2, detection_path1, detection_path2, options)

    #video_path1 = './videos/my_video/cam1_right_video1_part_3.avi'
    #video_path2 = './videos/my_video/cam1_right_video2_part_3.avi'
    #detection_path1 = './videos/detections/detection_cam1_right_video1_part_3_cam1.txt'
    #detection_path2 = './videos/detections/detection_cam1_right_video2_part_3_cam2.txt'
    #obj_tracker.multi_tracking_object_read(video_path1, video_path2, detection_path1, detection_path2, options)

    #video_path1 = './videos/my_video/cam1_right_video1_part_4.avi'
    #video_path2 = './videos/my_video/cam1_right_video2_part_4.avi'
    #detection_path1 = './videos/detections/detection_cam1_right_video1_part_4_cam1.txt'
    #detection_path2 = './videos/detections/detection_cam1_right_video2_part_4_cam2.txt'
    #obj_tracker.multi_tracking_object_read(video_path1, video_path2, detection_path1, detection_path2, options)

    #video_path1 = './videos/my_video/cam1_right_video1_part_5.avi'
    #video_path2 = './videos/my_video/cam1_right_video2_part_5.avi'
    #detection_path1 = './videos/detections/detection_cam1_right_video1_part_5_cam1.txt'
    #detection_path2 = './videos/detections/detection_cam1_right_video2_part_5_cam2.txt'
    #obj_tracker.multi_tracking_object_read(video_path1, video_path2, detection_path1, detection_path2, options)

    # dinh's camera
    #video_path1 = './videos/my_video/cam1_left_video1_part_1.avi'
    #video_path2 = './videos/my_video/cam1_left_video2_part_1.avi'
    #detection_path1 = './videos/detections/detection_cam1_left_video1_part_1_cam1.txt'
    #detection_path2 = './videos/detections/detection_cam1_left_video2_part_1_cam2.txt'
    #obj_tracker.multi_tracking_object_read(video_path1, video_path2, detection_path1, detection_path2, options)

    #video_path1 = './videos/my_video/cam1_left_video1_part_2.avi'
    #video_path2 = './videos/my_video/cam1_left_video2_part_2.avi'
    #detection_path1 = './videos/detections/detection_cam1_left_video1_part_2_cam1.txt'
    #detection_path2 = './videos/detections/detection_cam1_left_video2_part_2_cam2.txt'
    #obj_tracker.multi_tracking_object_read(video_path1, video_path2, detection_path1, detection_path2, options)

    #video_path1 = './videos/my_video/cam1_left_video1_part_3.avi'
    #video_path2 = './videos/my_video/cam1_left_video2_part_3.avi'
    #detection_path1 = './videos/detections/detection_cam1_left_video1_part_3_cam1.txt'
    #detection_path2 = './videos/detections/detection_cam1_left_video2_part_3_cam2.txt'
    #obj_tracker.multi_tracking_object_read(video_path1, video_path2, detection_path1, detection_path2, options)

    #video_path1 = './videos/my_video/cam1_left_video1_part_4.avi'
    #video_path2 = './videos/my_video/cam1_left_video2_part_4.avi'
    #detection_path1 = './videos/detections/detection_cam1_left_video1_part_4_cam1.txt'
    #detection_path2 = './videos/detections/detection_cam1_left_video2_part_4_cam2.txt'
    #obj_tracker.multi_tracking_object_read(video_path1, video_path2, detection_path1, detection_path2, options)

    #video_path1 = './videos/my_video/cam1_left_video1_part_5.avi'
    #video_path2 = './videos/my_video/cam1_left_video2_part_5.avi'
    #detection_path1 = './videos/detections/detection_cam1_left_video1_part_5_cam1.txt'
    #detection_path2 = './videos/detections/detection_cam1_left_video2_part_5_cam2.txt'
    #obj_tracker.multi_tracking_object_read(video_path1, video_path2, detection_path1, detection_path2, options)

    # new version
    video_path1 = './videos/my_video/cam1_right_part_1.avi'
    video_path2 = './videos/my_video/cam2_right_part_1.avi'
    detection_path1 = './videos/detections/new_verson_parallel/detection_cam1_right_part_1_cam1.txt'
    detection_path2 = './videos/detections/new_verson_parallel/detection_cam2_right_part_1_cam2.txt'
    obj_tracker.multi_tracking_object_read(video_path1, video_path2, detection_path1, detection_path2, options)

    video_path1 = './videos/my_video/cam1_right_part_2.avi'
    video_path2 = './videos/my_video/cam2_right_part_2.avi'
    detection_path1 = './videos/detections/new_verson_parallel/detection_cam1_right_part_2_cam1.txt'
    detection_path2 = './videos/detections/new_verson_parallel/detection_cam2_right_part_2_cam2.txt'
    obj_tracker.multi_tracking_object_read(video_path1, video_path2, detection_path1, detection_path2, options)

    video_path1 = './videos/my_video/cam1_right_part_3.avi'
    video_path2 = './videos/my_video/cam2_right_part_3.avi'
    detection_path1 = './videos/detections/new_verson_parallel/detection_cam1_right_part_3_cam1.txt'
    detection_path2 = './videos/detections/new_verson_parallel/detection_cam2_right_part_3_cam2.txt'
    obj_tracker.multi_tracking_object_read(video_path1, video_path2, detection_path1, detection_path2, options)

    video_path1 = './videos/my_video/cam1_right_part_4.avi'
    video_path2 = './videos/my_video/cam2_right_part_4.avi'
    detection_path1 = './videos/detections/new_verson_parallel/detection_cam1_right_part_4_cam1.txt'
    detection_path2 = './videos/detections/new_verson_parallel/detection_cam2_right_part_4_cam2.txt'
    obj_tracker.multi_tracking_object_read(video_path1, video_path2, detection_path1, detection_path2, options)

    video_path1 = './videos/my_video/cam1_right_part_5.avi'
    video_path2 = './videos/my_video/cam2_right_part_5.avi'
    detection_path1 = './videos/detections/new_verson_parallel/detection_cam1_right_part_5_cam1.txt'
    detection_path2 = './videos/detections/new_verson_parallel/detection_cam2_right_part_5_cam2.txt'
    obj_tracker.multi_tracking_object_read(video_path1, video_path2, detection_path1, detection_path2, options)

    video_path1 = './videos/my_video/cam1_right_part_6.avi'
    video_path2 = './videos/my_video/cam2_right_part_6.avi'
    detection_path1 = './videos/detections/new_verson_parallel/detection_cam1_right_part_6_cam1.txt'
    detection_path2 = './videos/detections/new_verson_parallel/detection_cam2_right_part_6_cam2.txt'
    obj_tracker.multi_tracking_object_read(video_path1, video_path2, detection_path1, detection_path2, options)

def write(options):
    video_path1 = './videos/my_video/cam1_right_part_1.avi'
    video_path2 = './videos/my_video/cam2_right_part_1.avi'
    obj_tracker.multi_tracking_object_write(video_path1, video_path2, options)

    video_path1 = './videos/my_video/cam1_right_part_2.avi'
    video_path2 = './videos/my_video/cam2_right_part_2.avi'
    obj_tracker.multi_tracking_object_write(video_path1, video_path2, options)

    video_path1 = './videos/my_video/cam1_right_part_3.avi'
    video_path2 = './videos/my_video/cam2_right_part_3.avi'
    obj_tracker.multi_tracking_object_write(video_path1, video_path2, options)

    video_path1 = './videos/my_video/cam1_right_part_4.avi'
    video_path2 = './videos/my_video/cam2_right_part_4.avi'
    obj_tracker.multi_tracking_object_write(video_path1, video_path2, options)

    video_path1 = './videos/my_video/cam1_right_part_5.avi'
    video_path2 = './videos/my_video/cam2_right_part_5.avi'
    obj_tracker.multi_tracking_object_write(video_path1, video_path2, options)

    video_path1 = './videos/my_video/cam1_right_part_6.avi'
    video_path2 = './videos/my_video/cam2_right_part_6.avi'
    obj_tracker.multi_tracking_object_write(video_path1, video_path2, options)

    video_path1 = './videos/sample_video/campusc7_c0_edit.avi'
    video_path2 = './videos/sample_video/campusc7_c1_edit.avi'
    obj_tracker.multi_tracking_object_write(video_path1, video_path2, options)

def live(options):
    obj_tracker.tracking_object(video_path1, video_path2, options)
    obj_tracker.tracking_single_camera('./videos/video2_1.avi', options)
    obj_tracker.multi_tracking_object(video_path1, video_path2, options)

if __name__ == "__main__":
    #live(options)
    #write(options)
    read(options)


