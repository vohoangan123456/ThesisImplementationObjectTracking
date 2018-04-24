'''
    File name         : object_tracking.py
    File Description  : Multi Objects Tracker Using Yolo & Kalman Filter Algorithm
    Author            : An Vo Hoang
    Date created      : 10/02/2018
    Date last modified: 10/02/2018
    Python Version    : 3.6
'''

# Import python libraries
import cv2
import copy
import os
from random import randint
from my_working_space.kalman_filter.yolo_detector import yolo_detector
from my_working_space.kalman_filter.tracker import Tracker
#from my_working_space.kalman_filter.multi_tracker import MultiTracker, Tracker
from darkflow.net.build import TFNet


def tracking_object(videopath1, videopath2, options):
    rd_number = randint(0, 100) # random number for save video

    # init darkflow network
    tfNet = TFNet(options) 
    
    # Create opencv video capture object
    cam1 = cv2.VideoCapture(videopath1)
    cam2 = cv2.VideoCapture(videopath2)

    frame_width, frame_height = 600,600
    # create write video
    #frame_width = int(cam1.get(3))
    #frame_height = int(cam1.get(4))
    out1 = cv2.VideoWriter('./videos/outpy_{0}_{1}'.format(str(rd_number), os.path.basename(videopath1)),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    #frame_width = int(cam2.get(3))
    #frame_height = int(cam2.get(4))
    out2 = cv2.VideoWriter('./videos/outpy_{0}_{1}'.format(str(rd_number), os.path.basename(videopath2)),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    # Create Object Detector
    yolo_detectors1 = yolo_detector(tfNet)
    yolo_detectors2 = yolo_detector(tfNet)

    # Create Object Tracker
    tracker1 = Tracker(320, 10, 5, 100, 1)
    tracker2 = Tracker(320, 10, 5, 300, 2)

    firstFrame = True   # flag check we capture the first frame

    # Infinite loop to process video frames
    while(True):
        # Capture frame-by-frame
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        if frame1 is None or frame2 is None:
            break
        frame1 = cv2.resize(frame1, (600,600))
        frame2 = cv2.resize(frame2, (600,600))

        # Make copy of original frame
        orig_frame1 = copy.copy(frame1)
        orig_frame2 = copy.copy(frame2)

        if firstFrame is True:
            tracker1.get_FOV(orig_frame2, orig_frame1)
            tracker2.get_FOV(orig_frame1, orig_frame2)        

        # Detect and return centeroids of the objects in the frame
        yolo_detectors1.detect(frame1)
        yolo_detectors2.detect(frame2)
        
        # draw fov
        tracker1.fov.draw_polygon(frame1)
        tracker2.fov.draw_polygon(frame2)

        # If centroids are detected then track them
        # Track object using Kalman Filter
        # cam1
        #if (len(yolo_detectors1.list_moving_obj) > 0):        
        tracker1.Update(yolo_detectors1.list_moving_obj)
        #if firstFrame is True:
        #    tracker2.assign_label_second_camera_first_frame(yolo_detectors2.list_moving_obj, tracker1)
        # cam2
        #if (len(yolo_detectors2.list_moving_obj) > 0):
        tracker2.Update(yolo_detectors2.list_moving_obj)            

        # if there is any moving object that were not assigned but exist inside the FOV, need to assign by the appropriate object in another camera
        #if firstFrame is not True:
        tracker1.Update_un_assign_detect(yolo_detectors1.list_moving_obj, tracker2)
        tracker2.Update_un_assign_detect(yolo_detectors2.list_moving_obj, tracker1)

        # For identified object tracks draw tracking line
        # cam1
        for i in range(len(tracker1.tracks)):
            if tracker1.tracks[i].skipped_frames == 0:
                moving_obj_track = tracker1.tracks[i].moving_obj
                pX = moving_obj_track.bounding_box.pX
                pY = moving_obj_track.bounding_box.pY
                cv2.putText(frame1, str(tracker1.tracks[i].track_id % 100), (int(pX), int(pY)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
        cv2.imshow('Tracking1', frame1)
        # cam2
        for i in range(len(tracker2.tracks)):
            if tracker2.tracks[i].skipped_frames == 0:
                moving_obj_track = tracker2.tracks[i].moving_obj
                pX = moving_obj_track.bounding_box.pX
                pY = moving_obj_track.bounding_box.pY
                cv2.putText(frame2, str(tracker2.tracks[i].track_id % 100), (int(pX), int(pY)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
        cv2.imshow('Tracking2', frame2)
        if firstFrame is True:
            firstFrame = False
        #save frame to video
        out1.write(frame1)
        out2.write(frame2)

        # Display the original frame
        cv2.imshow('Original1', orig_frame1)
        cv2.imshow('Original2', orig_frame2)

        # Slower the FPS
        cv2.waitKey(50)

        # Check for key strokes
        k = cv2.waitKey(50) & 0xff
        if k == 27:  # 'esc' key has been pressed, exit program.
            break

    # When everything done, release the capture
    cam1.release()
    cam2.release()
    out1.release()
    out2.release()
    cv2.destroyAllWindows()

def tracking_single_camera(videopath, options):
    rd_number = randint(0, 100) # random number for save video

    # init darkflow network
    tfNet = TFNet(options) 
    
    # Create opencv video capture object
    cam1 = cv2.VideoCapture(videopath)

    frame_width, frame_height = 600,600
    # create write video
    out1 = cv2.VideoWriter('./videos/outpy_{0}_{1}'.format(str(rd_number), os.path.basename(videopath)),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    # Create Object Detector
    yolo_detectors1 = yolo_detector(tfNet)

    # Create Object Tracker
    tracker1 = Tracker(160, 25, 5, 100, 1)

    firstFrame = True   # flag check we capture the first frame

    # Infinite loop to process video frames
    while(True):
        # Capture frame-by-frame
        ret1, frame1 = cam1.read()
        if frame1 is None:
            break
        frame1 = cv2.resize(frame1, (600,600))

        # Make copy of original frame
        orig_frame1 = copy.copy(frame1)
        #orig_frame2 = cv2.imread('./sample_img/test2.jpg')
        #if firstFrame is True:
        #    firstFrame = False
        #    tracker1.get_FOV(orig_frame2, orig_frame1)    

        # Detect and return centeroids of the objects in the frame
        yolo_detectors1.detect(frame1)
        
        # draw fov
        #tracker1.fov.draw_polygon(frame1)
        
        tracker1.update_cam1(yolo_detectors1.list_moving_obj)         
        
        tracker1.update_single_camera(yolo_detectors1.list_moving_obj)

        # For identified object tracks draw tracking line
        # cam1
        for i in range(len(tracker1.tracks)):
            if tracker1.tracks[i].skipped_frames == 0:
                moving_obj_track = tracker1.tracks[i].moving_obj
                pX = moving_obj_track.bounding_box.pX
                pY = moving_obj_track.bounding_box.pY
                cv2.putText(frame1, str(tracker1.tracks[i].track_id % 100), (int(pX), int(pY)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
        cv2.imshow('Tracking1', frame1)
        #save frame to video
        out1.write(frame1)

        # Display the original frame
        cv2.imshow('Original1', orig_frame1)

        # Slower the FPS
        cv2.waitKey(50)

        # Check for key strokes
        k = cv2.waitKey(50) & 0xff
        if k == 27:  # 'esc' key has been pressed, exit program.
            break

    # When everything done, release the capture
    cam1.release()
    out1.release()
    cv2.destroyAllWindows()

def multi_tracking_object(videopath1, videopath2, options):
    rd_number = randint(0, 100) # random number for save video

    # init darkflow network
    tfNet = TFNet(options) 
    
    # Create opencv video capture object
    cam1 = cv2.VideoCapture(videopath1)
    cam2 = cv2.VideoCapture(videopath2)

    frame_width, frame_height = 600,600
    # create write video
    out1 = cv2.VideoWriter('./videos/outpy_{0}_{1}'.format(str(rd_number), os.path.basename(videopath1)),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    out2 = cv2.VideoWriter('./videos/outpy_{0}_{1}'.format(str(rd_number), os.path.basename(videopath2)),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    # Create Object Detector
    yolo_detectors1 = yolo_detector(tfNet)
    yolo_detectors2 = yolo_detector(tfNet)

    # Create Object Tracker
    tracker1 = Tracker(320, 10, 5, 100, 1)
    tracker2 = Tracker(320, 10, 5, 300, 2)

    firstFrame = True   # flag check we capture the first frame

    # Infinite loop to process video frames
    while(True):
        # Capture frame-by-frame
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        if frame1 is None or frame2 is None:
            break
        frame1 = cv2.resize(frame1, (600,600))
        frame2 = cv2.resize(frame2, (600,600))

        # Make copy of original frame
        orig_frame1 = copy.copy(frame1)
        orig_frame2 = copy.copy(frame2)

        if firstFrame is True:
            tracker1.get_FOV(orig_frame2, orig_frame1)
            tracker2.get_FOV(orig_frame1, orig_frame2)        

        # Detect and return centeroids of the objects in the frame
        yolo_detectors1.detect(frame1)
        yolo_detectors2.detect(frame2)
        
        # draw fov
        tracker1.fov.draw_polygon(frame1)
        tracker2.fov.draw_polygon(frame2)

        # If centroids are detected then track them
        # Track object using Kalman Filter
        # cam1
        #if (len(yolo_detectors1.list_moving_obj) > 0):        
        tracker1.Update(yolo_detectors1.list_moving_obj)
        tracker2.set_trackIdCount(tracker1.trackIdCount)
        # cam2
        #if (len(yolo_detectors2.list_moving_obj) > 0):
        tracker2.Update(yolo_detectors2.list_moving_obj)
        tracker1.set_trackIdCount(tracker2.trackIdCount)

        # assign label consistent
        tracker1.Update_un_assign_detect(yolo_detectors1.list_moving_obj, tracker2)
        tracker2.set_trackIdCount(tracker1.trackIdCount)

        tracker2.Update_un_assign_detect(yolo_detectors2.list_moving_obj, tracker1)
        tracker1.set_trackIdCount(tracker2.trackIdCount)
        #MultiTracker.consistent_label(tracker1, tracker2)

        # For identified object tracks draw tracking line
        # cam1
        for i in range(len(tracker1.tracks)):
            if tracker1.tracks[i].skipped_frames == 0:
                moving_obj_track = tracker1.tracks[i].moving_obj
                pX = moving_obj_track.bounding_box.pX
                pY = moving_obj_track.bounding_box.pY
                cv2.putText(frame1, str(tracker1.tracks[i].track_id % 100), (int(pX), int(pY)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
        cv2.imshow('Tracking1', frame1)
        # cam2
        for i in range(len(tracker2.tracks)):
            if tracker2.tracks[i].skipped_frames == 0:
                moving_obj_track = tracker2.tracks[i].moving_obj
                pX = moving_obj_track.bounding_box.pX
                pY = moving_obj_track.bounding_box.pY
                cv2.putText(frame2, str(tracker2.tracks[i].track_id % 100), (int(pX), int(pY)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
        cv2.imshow('Tracking2', frame2)
        if firstFrame is True:
            firstFrame = False
        #save frame to video
        out1.write(frame1)
        out2.write(frame2)

        # Display the original frame
        cv2.imshow('Original1', orig_frame1)
        cv2.imshow('Original2', orig_frame2)

        # Slower the FPS
        cv2.waitKey(50)

        # Check for key strokes
        k = cv2.waitKey(50) & 0xff
        if k == 27:  # 'esc' key has been pressed, exit program.
            break

    # When everything done, release the capture
    cam1.release()
    cam2.release()
    out1.release()
    out2.release()
    cv2.destroyAllWindows()