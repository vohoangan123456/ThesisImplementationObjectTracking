'''
    File name         : object_tracking.py
    File Description  : Multi Object Tracker Using Kalman Filter
                        and Hungarian Algorithm
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import cv2
import copy
import os
from random import randint
from my_working_space.kalman_filter.yolo_detector import yolo_detector
from my_working_space.kalman_filter.tracker import Tracker
from darkflow.net.build import TFNet

def tracking_object(videopath, options):
    tfNet = TFNet(options)
    # Create opencv video capture object
    cap = cv2.VideoCapture(videopath)
    # create write video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('./videos/outpy_{0}_{1}.avi'.format(os.path.basename(videopath), str(randint(0, 100))),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    # Create Object Detector
    yolo_detectors = yolo_detector(tfNet)

    # Create Object Tracker
    tracker = Tracker(160, 10, 5, 100)

    # Variables initialization
    skip_frame_count = 0
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
    pause = False

    # Infinite loop to process video frames
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        #frame = cv2.resize(frame, (600, 600))

        # Make copy of original frame
        orig_frame = copy.copy(frame)

        # Skip initial frames that display logo
        if (skip_frame_count < 15):
            skip_frame_count += 1
            continue

        # Detect and return centeroids of the objects in the frame
        #centers = detector.Detect(frame)
        centers = yolo_detectors.detect(frame)

        # If centroids are detected then track them
        if (len(centers) > 0):

            # Track object using Kalman Filter
            tracker.Update(centers)

            # For identified object tracks draw tracking line
            # Use various colors to indicate different track_id
            for i in range(len(tracker.tracks)):
                if (len(tracker.tracks[i].trace) > 1):
                    for j in range(len(tracker.tracks[i].trace)-1):
                        # Draw trace line
                        x1 = tracker.tracks[i].trace[j][0][0]
                        y1 = tracker.tracks[i].trace[j][1][0]
                        x2 = tracker.tracks[i].trace[j+1][0][0]
                        y2 = tracker.tracks[i].trace[j+1][1][0]
                        clr = tracker.tracks[i].track_id % 9
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                 track_colors[clr], 2)

            # Display the resulting tracking frame
            cv2.imshow('Tracking', frame)
        out.write(frame)
        # Display the original frame
        cv2.imshow('Original', orig_frame)

        # Slower the FPS
        cv2.waitKey(50)

        # Check for key strokes
        k = cv2.waitKey(50) & 0xff
        if k == 27:  # 'esc' key has been pressed, exit program.
            break
        if k == 112:  # 'p' has been pressed. this will pause/resume the code.
            pause = not pause
            if (pause is True):
                print("Code is paused. Press 'p' to resume..")
                while (pause is True):
                    # stay in this loop until
                    key = cv2.waitKey(30) & 0xff
                    if key == 112:
                        pause = False
                        print("Resume code..!!")
                        break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()