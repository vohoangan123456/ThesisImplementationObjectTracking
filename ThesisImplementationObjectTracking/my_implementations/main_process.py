'''
    File name         : main_process.py
    File Description  : handle the thesis process
    Author            : An Vo
    Date created      : 19/04/2018
    Python Version    : 3.6
'''
import cv2
import copy
import os
from random import randint
from darkflow.net.build import TFNet
from my_implementations.object_tracking.tracker import Tracker
from my_implementations.common.global_config import *
from my_implementations.utils.utils import merge_two_video

def multi_tracking_object(videopath1, videopath2, options):
    '''
        Description:
            multiple object tracking with live yolo detector
        Params:
            videopath1(str): the path of the first video
            videopath2(str): the path of the second video
            options: the setting options oftensorflow network
    '''
    from my_implementations.object_detection.detectors import yolo_detector
    rd_number = randint(0, 1000)

    # init darkflow network
    tfNet = TFNet(options) 
    
    # Create opencv video capture object
    cam1 = cv2.VideoCapture(videopath1)
    cam2 = cv2.VideoCapture(videopath2)

    frame_width, frame_height = 600,600
    file_path1 = './videos/outpy_{0}_{1}'.format(str(rd_number), os.path.basename(videopath1))
    file_path2 = './videos/outpy_{0}_{1}'.format(str(rd_number), os.path.basename(videopath2))
    # create write video
    out1 = cv2.VideoWriter(file_path1,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    out2 = cv2.VideoWriter(file_path2,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    # Create Object Detector
    yolo_detectors1 = yolo_detector(tfNet)
    yolo_detectors2 = yolo_detector(tfNet)

    # Create Object Tracker
    tracker1 = Tracker(0.85, 20, 5, 100, 1)
    tracker2 = Tracker(0.85, 20, 5, 100, 2)


    firstFrame = True   # flag check we capture the first frame
    frame_index = 0
    # Infinite loop to process video frames
    while(True):
        # Capture frame-by-frame
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        #frame_index += 1
        #if frame_index < 1300:
        #    continue
        if frame1 is None or frame2 is None:
            break
        frame1 = cv2.resize(frame1, (600,600))
        frame2 = cv2.resize(frame2, (600,600))

        # Make copy of original frame
        orig_frame1 = copy.copy(frame1)
        orig_frame2 = copy.copy(frame2)

        if firstFrame is True:
            tracker1.get_FOV(orig_frame2, orig_frame1, FOV_OF_CAM2_IN_CAM1)
            tracker2.get_FOV(orig_frame1, orig_frame2, FOV_OF_CAM1_IN_CAM2)

        # Detect and return centeroids of the objects in the frame
        yolo_detectors1.detect(frame1)
        yolo_detectors2.detect(frame2)
        
        # draw fov
        tracker1.fov.draw_polygon(frame1)
        tracker2.fov.draw_polygon(frame2)

        # track moving object in
        # cam1
        tracker1.Update(yolo_detectors1.list_moving_obj)
        tracker2.set_trackId(tracker1.trackId)
        # cam2
        tracker2.Update(yolo_detectors2.list_moving_obj)
        tracker1.set_trackId(tracker2.trackId)

        # assign label for unassigned moving object
        if firstFrame is True:
            tracker1.assign_label_consistent(tracker2)
            tracker2.set_trackId(tracker1.trackId)
        else:
            tracker1.Update_un_assign_detect(yolo_detectors1.list_moving_obj, tracker2)
            tracker2.set_trackId(tracker1.trackId)

            tracker2.Update_un_assign_detect(yolo_detectors2.list_moving_obj, tracker1)
            tracker1.set_trackId(tracker2.trackIdCount)

        # For identified object tracks draw tracking line
        # cam1
        for i in range(len(tracker1.tracks)):
            if tracker1.tracks[i].skipped_frames == 0:
                moving_obj_track = tracker1.tracks[i].moving_obj
                pX = moving_obj_track.bounding_box.pX
                pY = moving_obj_track.bounding_box.pY
                tl = (int(pX), int(pY))
                br = (int(pX + moving_obj_track.bounding_box.width), int(pY + moving_obj_track.bounding_box.height))
                cv2.rectangle(frame1, tl, br, (0, 255, 0), 1)
                cv2.putText(frame1, str(tracker1.tracks[i].track_id % 100), tl, cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
                cv2.putText(frame1, str('({0},{1})'.format(tl[0], br[1])), (tl[0], br[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                if moving_obj_track.is_in_fov is True:
                    cv2.putText(frame1, 'in', (tl[0] + 10, tl[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                if moving_obj_track.bounding_box.is_under_of_occlusion is True:
                    position = 'b-r';
                    if moving_obj_track.bounding_box.is_topleft_occlusion is True:
                        position = 't-l'
                    cv2.putText(frame1, '{0}:{1}'.format(position, str(moving_obj_track.bounding_box.overlap_percent)), (tl[0] + 10, tl[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow('Tracking1', frame1)
        # cam2
        for i in range(len(tracker2.tracks)):
            if tracker2.tracks[i].skipped_frames == 0:
                moving_obj_track = tracker2.tracks[i].moving_obj
                pX = moving_obj_track.bounding_box.pX
                pY = moving_obj_track.bounding_box.pY
                tl = (int(pX), int(pY))
                br = (int(pX + moving_obj_track.bounding_box.width), int(pY + moving_obj_track.bounding_box.height))
                cv2.rectangle(frame2, tl, br, (0, 255, 0), 1)
                cv2.putText(frame2, str(tracker2.tracks[i].track_id % 100), tl, cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
                cv2.putText(frame2, str('({0},{1})'.format(tl[0], br[1])), (tl[0], br[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                if moving_obj_track.is_in_fov is True:
                    cv2.putText(frame2, 'in', (tl[0] + 10, tl[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                if moving_obj_track.bounding_box.is_under_of_occlusion is True:
                    position = 'b-r';
                    if moving_obj_track.bounding_box.is_topleft_occlusion is True:
                        position = 't-l'
                    cv2.putText(frame2, '{0}:{1}'.format(position, str(moving_obj_track.bounding_box.overlap_percent)), (tl[0] + 10, tl[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
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
        cv2.waitKey(18)

        # Check for key strokes
        k = cv2.waitKey(18) & 0xff
        if k == 27:  # 'esc' key has been pressed, exit program.
            break    
    # When everything done, release the capture
    cam1.release()
    cam2.release()
    out1.release()
    out2.release()
    cv2.destroyAllWindows()
    merge_two_video(file_path1, file_path2)

def multi_tracking_object_write(videopath1, videopath2, options):
    from my_implementations.object_detection.detectors import yolo_detector_write
    rd_number = randint(0, 1000) # random number for save video
    # init darkflow network
    tfNet = TFNet(options) 
    
    # Create opencv video capture object
    cam1 = cv2.VideoCapture(videopath1)
    cam2 = cv2.VideoCapture(videopath2)
    # Create Object Detector
    yolo_detectors1 = yolo_detector_write(tfNet, 1, rd_number)
    yolo_detectors2 = yolo_detector_write(tfNet, 2, rd_number)

    frame_index = 0
    # Infinite loop to process video frames
    while(True):
        # Capture frame-by-frame
        ret1, frame1  = cam1.read()
        ret2, frame2 = cam2.read()
        frame_index += 1
        if frame1 is None or frame2 is None:
            break

        # Detect and return centeroids of the objects in the frame
        yolo_detectors1.detect(frame1)
        yolo_detectors2.detect(frame2)
        # Display the original frame
        cv2.imshow('Original1', frame1)
        cv2.imshow('Original2', frame2)

        # Slower the FPS
        cv2.waitKey(18)

        # Check for key strokes
        k = cv2.waitKey(18) & 0xff
        if k == 27:  # 'esc' key has been pressed, exit program.
            break    
    # When everything done, release the capture
    yolo_detectors1.release_file()
    yolo_detectors2.release_file()
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()

def multi_tracking_object_read(videopath1, videopath2, filedetector1, filedetector2, options):
	from my_implementations.object_detection.detectors import yolo_detector_read
	rd_number = randint(0, 1000) # random number for save video

	# Create opencv video capture object
	cam1 = cv2.VideoCapture(videopath1)
	cam2 = cv2.VideoCapture(videopath2)

	frame_width1 = 600 #int(cam1.get(3))
	frame_height1 = 600 #int(cam1.get(4))
	file_path1 = './videos/outpy_{0}_{1}'.format(str(rd_number), os.path.basename(videopath1))
	file_path2 = './videos/outpy_{0}_{1}'.format(str(rd_number), os.path.basename(videopath2))
	# create write video
	out1 = cv2.VideoWriter(file_path1,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width1,frame_height1))
	frame_width2 = 600 #int(cam2.get(3))
	frame_height2 = 600 #int(cam2.get(4))
	out2 = cv2.VideoWriter(file_path2,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width2,frame_height2))

	# Create Object Detector
	yolo_detectors1 = yolo_detector_read(filedetector1)
	yolo_detectors2 = yolo_detector_read(filedetector2)

	# Create Object Tracker
	tracker1 = Tracker(190, 30, 5, 100, 1)
	tracker2 = Tracker(190, 30, 5, 100, 2)


	firstFrame = True   # flag check we capture the first frame
	frame_index = 0
	# Infinite loop to process video frames
	while(True):
		#ret1, frame1 = cam1.read()
		# Capture frame-by-frame
		ret1, frame1 = cam1.read()
		ret2, frame2 = cam2.read()
		frame_index += 1
		if frame1 is None or frame2 is None:
			break
		frame1 = cv2.resize(frame1, (600, 600))
		frame2 = cv2.resize(frame2, (600, 600))
		# Make copy of original frame
		orig_frame1 = copy.copy(frame1)
		orig_frame2 = copy.copy(frame2)

		if firstFrame is True:
			tracker1.get_FOV(orig_frame2, orig_frame1, FOV_OF_CAM2_IN_CAM1)
			tracker2.get_FOV(orig_frame1, orig_frame2, FOV_OF_CAM1_IN_CAM2)

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
		tracker2.set_trackId(tracker1.trackId)
		# cam2
		#if (len(yolo_detectors2.list_moving_obj) > 0):
		tracker2.Update(yolo_detectors2.list_moving_obj)
		tracker1.set_trackId(tracker2.trackId)

		# assign label consistent
		if firstFrame is True:
			tracker1.assign_label_consistent(tracker2)
			tracker2.set_trackId(tracker1.trackId)
		else:
			tracker1.Update_un_assign_detect(yolo_detectors1.list_moving_obj, tracker2)
			tracker2.set_trackId(tracker1.trackId)

			tracker2.Update_un_assign_detect(yolo_detectors2.list_moving_obj, tracker1)
			tracker1.set_trackId(tracker2.trackId)
		#MultiTracker.consistent_label(tracker1, tracker2)

        # For identified object tracks draw tracking line
        # cam1
		for i in range(len(tracker1.tracks)):
			if tracker1.tracks[i].skipped_frames == 0:
				moving_obj_track = tracker1.tracks[i].moving_obj
				pX = moving_obj_track.bounding_box.pX
				pY = moving_obj_track.bounding_box.pY
				tl = (int(pX), int(pY))
				br = (int(pX + moving_obj_track.bounding_box.width), int(pY + moving_obj_track.bounding_box.height))
				cv2.rectangle(frame1, tl, br, (0, 255, 0), 2)
				cv2.putText(frame1, str(tracker1.tracks[i].track_id % 100), tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
				cv2.putText(frame1, str('({0},{1})'.format(tl[0], br[1])), (tl[0], br[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
				if moving_obj_track.is_in_fov is True:
					cv2.putText(frame1, 'in', (tl[0] + 10, tl[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
				if moving_obj_track.bounding_box.is_under_of_occlusion is True:
					position = 'b-r';
					if moving_obj_track.bounding_box.is_topleft_occlusion is True:
						position = 't-l'
					cv2.putText(frame1, '{0}:{1}'.format(position, str(moving_obj_track.bounding_box.overlap_percent)), (tl[0] + 10, tl[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
				if moving_obj_track.bounding_box.is_disappear is True:
					cv2.putText(frame1, 'dis', (tl[0] + 10, tl[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
		cv2.imshow('Tracking1', frame1)
        # cam2
		for i in range(len(tracker2.tracks)):
			if tracker2.tracks[i].skipped_frames == 0:
				moving_obj_track = tracker2.tracks[i].moving_obj
				pX = moving_obj_track.bounding_box.pX
				pY = moving_obj_track.bounding_box.pY
				tl = (int(pX), int(pY))
				br = (int(pX + moving_obj_track.bounding_box.width), int(pY + moving_obj_track.bounding_box.height))
				cv2.rectangle(frame2, tl, br, (0, 255, 0), 2)
				cv2.putText(frame2, str(tracker2.tracks[i].track_id % 100), tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
				cv2.putText(frame2, str('({0},{1})'.format(tl[0], br[1])), (tl[0], br[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
				if moving_obj_track.is_in_fov is True:
					cv2.putText(frame2, 'in', (tl[0] + 10, tl[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
				if moving_obj_track.bounding_box.is_under_of_occlusion is True:
					position = 'b-r';
					if moving_obj_track.bounding_box.is_topleft_occlusion is True:
						position = 't-l'
					cv2.putText(frame2, '{0}:{1}'.format(position, str(moving_obj_track.bounding_box.overlap_percent)), (tl[0] + 10, tl[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
				if moving_obj_track.bounding_box.is_disappear is True:
					cv2.putText(frame1, 'dis', (tl[0] + 10, tl[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
		cv2.imshow('Tracking2', frame2)
		if firstFrame is True:
			firstFrame = False
		#save frame to video
		out1.write(frame1)
		out2.write(frame2)

        ## Display the original frame
        #cv2.imshow('Original1', orig_frame1)
        #cv2.imshow('Original2', orig_frame2)

        # Slower the FPS
		cv2.waitKey(18)

        # Check for key strokes
		k = cv2.waitKey(18) & 0xff
		if k == 27:  # 'esc' key has been pressed, exit program.
			break    
    # When everything done, release the capture
	cam1.release()
	cam2.release()
	out1.release()
	out2.release()
	cv2.destroyAllWindows()
	merge_two_video(file_path1, file_path2)

#def multi_tracking_object_read(videopath1, videopath2, filedetector1, filedetector2, options):
#	from my_implementations.object_detection.detectors import yolo_detector_read
#	rd_number = randint(0, 1000)

#	cam1 = cv2.VideoCapture(videopath1)
#	cam2 = cv2.VideoCapture(videopath2)
#	frame_width1 = 600 #int(cam1.getestc(3))
#	frame_height1 = 600 #inext(cam1.getestc(4))
#	file_path1 = './videos/outpy_{0}_{1}'.format(str(rd_number), os.path.basename(videopath1))
#	file_path2 = './videos/outpy_{0}_{1}'.format(str(rd_number), os.path.basename(videopath2))

#	out1 = cv2.VideoWriter(file_path1,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width1,frame_height1))
#	frame_width2 = 600 #int(cam2.get(3))
#	frame_height2 = 600 #int(cam2.get(4))
#	out2 = cv2.VideoWriter(file_path2,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width2,frame_height2))
#	yolo_detectors1 = yolo_detector_read(filedetector1)
#	yolo_detectors2 = yolo_detector_read(filedetector2)