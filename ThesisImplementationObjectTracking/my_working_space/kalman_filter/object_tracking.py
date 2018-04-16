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
from darkflow.net.build import TFNet

def tracking_object(videopath1, videopath2, options):
    rd_number = randint(0, 100)
    tfNet = TFNet(options)
    # Create opencv video capture object
    cam1 = cv2.VideoCapture(videopath1)
    cam2 = cv2.VideoCapture(videopath2)

    # create write video
    frame_width = int(cam1.get(3))
    frame_height = int(cam1.get(4))
    out1 = cv2.VideoWriter('./videos/outpy_{0}_{1}.avi'.format(str(rd_number), os.path.basename(videopath1)),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    frame_width = int(cam2.get(3))
    frame_height = int(cam2.get(4))
    out2 = cv2.VideoWriter('./videos/outpy_{0}_{1}.avi'.format(str(rd_number), os.path.basename(videopath2)),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    # Create Object Detector
    yolo_detectors1 = yolo_detector(tfNet)
    yolo_detectors2 = yolo_detector(tfNet)

    # Create Object Tracker
    tracker1 = Tracker(160, 10, 5, 100)
    tracker2 = Tracker(160, 10, 5, 100)

    # Infinite loop to process video frames
    while(True):
        # Capture frame-by-frame
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()

        if frame1 is None or frame2 is None:
            break

        # Make copy of original frame
        orig_frame1 = copy.copy(frame1)
        orig_frame2 = copy.copy(frame2)

        # Detect and return centeroids of the objects in the frame
        yolo_detectors1.detect(frame1)
        yolo_detectors2.detect(frame2)

        # If centroids are detected then track them
        # cam1
        if (len(yolo_detectors1.list_moving_obj) > 0):
            # Track object using Kalman Filter
            tracker1.Update(yolo_detectors1.list_moving_obj)

            # For identified object tracks draw tracking line
            # Use various colors to indicate different track_id
            for i in range(len(tracker1.tracks)):
                if (len(tracker1.tracks[i].trace) > 1):
                    pX = tracker1.tracks[i].trace[-1][0][0]
                    pY = tracker1.tracks[i].trace[-1][1][0]
                    cv2.putText(frame1, str(tracker1.tracks[i].track_id - 99), (int(pX), int(pY)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Display the resulting tracking frame
            cv2.imshow('Tracking1', frame1)
        # cam2
        if (len(yolo_detectors2.list_moving_obj) > 0):
            tracker2.Update(yolo_detectors2.list_moving_obj)
            for i in range(len(tracker2.tracks)):
                if (len(tracker2.tracks[i].trace) > 1):
                    pX = tracker2.tracks[i].trace[-1][0][0]
                    pY = tracker2.tracks[i].trace[-1][1][0]
                    cv2.putText(frame2, str(tracker2.tracks[i].track_id - 99), (int(pX), int(pY)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Display the resulting tracking frame
            cv2.imshow('Tracking2', frame2)
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
