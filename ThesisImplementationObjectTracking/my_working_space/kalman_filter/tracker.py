'''
    File name         : tracker.py
    File Description  : Tracker Using Kalman Filter & Hungarian Algorithm
    Author            : An Vo Hoang
    Date created      : 10/02/2018
    Date last modified: 10/02/2018
    Python Version    : 3.6
'''

# Import python libraries
import numpy as np
import sys
from scipy.optimize import linear_sum_assignment
from my_working_space.kalman_filter.kalman_filter import KalmanFilter
from my_working_space.kalman_filter.moving_object import MovingObject
from my_working_space.kalman_filter.field_of_view import CommonFOV
from my_working_space.kalman_filter.common import stable_matching, get_fov_from_image, iou_compute, check_obj_disappear
from my_working_space.kalman_filter.sort_kalman import KalmanBoxTracker

THRESHOLD = 160
WEIGHT = [100,1]  # [diff_distance, diff_feature]

#class Track(object):
#    def __init__(self, trackIdCount, moving_obj):
#        self.moving_obj = moving_obj            # moving object that be tracked
#        self.moving_obj.set_label(trackIdCount)
#        self.track_id = trackIdCount            # identification of each track object
#        self.KF = KalmanFilter()                # KF instance to track this object
#        self.prediction = np.asarray(moving_obj.center)# predicted centroids (x,y)
#        self.skipped_frames = 0                 # number of frames skipped undetected
#        self.trace = []                         # trace path

class Track(object):
    def __init__(self, trackIdCount, moving_obj):
        self.moving_obj = moving_obj            # moving object that be tracked
        self.moving_obj.set_label(trackIdCount)
        self.track_id = trackIdCount            # identification of each track object
        self.KF = KalmanBoxTracker(moving_obj.bounding_box)                # KF instance to track this object
        self.prediction = np.asarray(moving_obj.center)# predicted centroids (x,y)
        self.skipped_frames = 0                 # number of frames skipped undetected
        self.under_occlusion_frame = 0          # number of frames under occlusion
        self.trace = []                         # trace path


class Tracker(object):
    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length, trackIdCount, camId):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount
        self.un_assigned_detects = []
        self.object_in_fov = []
        self.fov = CommonFOV()
        self.camera_id = camId

    def get_FOV(self, target_cam, source_cam, img_path = None):
        # init two FOV of two camera
        if img_path is not None:
            list_point = get_fov_from_image(img_path)
            self.fov.generate_fov_from_list_point(list_point)
        else:
            self.fov.get_FOV_of_target_in_source(target_cam, source_cam)
    def set_trackIdCount(self, trackIdCount):
        self.trackIdCount = trackIdCount
    def Update(self, list_moving_obj):
        """
        Args:
            list_moving_obj: detected moving objects to be tracked
        Return:
            None
        """

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(list_moving_obj)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(list_moving_obj)):
                try:
                    previousObj = self.tracks[i].moving_obj
                    bb_test = [previousObj.topLeft[0][0], previousObj.topLeft[1][0], previousObj.bottomRight[0][0], previousObj.bottomRight[1][0]]
                    bb_gt = [list_moving_obj[j].topLeft[0][0], list_moving_obj[j].topLeft[1][0], list_moving_obj[j].bottomRight[0][0], list_moving_obj[j].bottomRight[1][0]]
                    iou = iou_compute(bb_test, bb_gt)
                    distance = 1 - iou

                    pre_bbx = self.tracks[i].KF.get_state()
                    distances = np.sqrt((pre_bbx.pX - list_moving_obj[j].bounding_box.pX)**2 + (pre_bbx.pYmax - list_moving_obj[j].bounding_box.pYmax)**2)

                    cost[i][j] = WEIGHT[0] * distance + WEIGHT[1] * distances
                except:
                    pass

        # Let's average the squared ERROR
        #cost = (0.5) * cost
        # Using Hungarian Algorithm assign the correct detected measurements to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        print('# cost')
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                print('id: {0}; cost: {1}'.format(self.tracks[i].track_id, cost[i][assignment[i]]))
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if (cost[i][assignment[i]] < self.dist_thresh) or (cost[i][assignment[i]] < (self.dist_thresh + 30) and self.tracks[i].moving_obj.bounding_box.is_under_of_occlusion):
                    pass
                else:
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
            else:
                move_obj = self.tracks[i].moving_obj
                if len(self.tracks[i].trace) > 2 and (move_obj.bounding_box.is_under_of_occlusion or check_obj_disappear(move_obj.bounding_box, move_obj.img_full)):
                    # if the previous frame obj is under occlusion => update moving obj by predict value
                    move_obj.update_bbx()
                    self.tracks[i].under_occlusion_frame += 1
                else:
                    self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
            if (self.tracks[i].under_occlusion_frame > self.max_frames_to_skip):
                self.tracks[i].moving_obj.bounding_box.is_under_of_occlusion = False
                self.tracks[i].under_occlusion_frame = 0
                self.tracks[i].skipped_frames = 1
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                else:
                    print("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        self.un_assigned_detects = []
        self.object_in_fov = []
        un_assigned_detects_out_fov = []
        for i in range(len(list_moving_obj)):
            list_moving_obj[i].set_existed_in(self.camera_id)
            # check if the selected moving object is in the fov
            existed_in_fov = self.fov.check_moving_obj_inside_FOV(list_moving_obj[i])
            # add object in fov to list
            if existed_in_fov:
                self.get_position_in_fov(list_moving_obj[i])
                self.object_in_fov.append(list_moving_obj[i])    
                # compute the distance from moving object to 4 edge of fov
                list_moving_obj[i].distance_to_fov_edge(self.fov)
            if i not in assignment:
                # if the moving object is unassigned but it's inside FOV => it was assigned in the other camera
                if existed_in_fov:
                    self.un_assigned_detects.append(i)
                else:
                    un_assigned_detects_out_fov.append(i)

        # Start new tracks with the moving object that first appear in camera
        if(len(un_assigned_detects_out_fov) != 0):
            for i in range(len(un_assigned_detects_out_fov)):
                track = Track(self.trackIdCount, list_moving_obj[un_assigned_detects_out_fov[i]])
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].under_occlusion_frame = 0
                self.tracks[i].KF.update(list_moving_obj[assignment[i]].bounding_box)
                list_moving_obj[assignment[i]].set_label(self.tracks[i].track_id)
                self.tracks[i].moving_obj = list_moving_obj[assignment[i]]
            else:
                self.tracks[i].KF.kf.P = np.zeros(self.tracks[i].KF.kf.P.shape)
                if self.tracks[i].skipped_frames == 0:
                    # in the case the moving obj is under occlusion => predict by current state
                    if len(self.tracks[i].trace) > 0:
                        new_bbx_predict = self.tracks[i].KF.get_state()
                        old_bbx_predict = self.tracks[i].trace[-1]
                        dx = new_bbx_predict.pX - old_bbx_predict.pX
                        dy = new_bbx_predict.pY - old_bbx_predict.pY
                        dx = int(dx/max(abs(dx), 1))
                        dy = int(dy/max(abs(dy), 1))
                        self.tracks[i].moving_obj.bounding_box.pX += dx
                        self.tracks[i].moving_obj.bounding_box.pY += dy
                    self.tracks[i].KF.update(self.tracks[i].moving_obj.bounding_box)
                else:
                    self.tracks[i].KF.update(self.tracks[i].moving_obj.bounding_box)
            # update the predict attribute in moving object
            move_obj = self.tracks[i].moving_obj
            new_bbx = self.tracks[i].KF.get_state()
            self.tracks[i].moving_obj.update_predict_bbx(new_bbx.pX, new_bbx.pY)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) - self.max_trace_length):
                    del self.tracks[i].trace[j]
            if self.tracks[i].under_occlusion_frame == 0:
                self.tracks[i].trace.append(new_bbx)
    def Update_un_assign_detect(self, list_moving_obj, another_tracker):
        '''
            Description:
                assign for unassigned moving object that exist inside the FOV
            Params:
                list_moving_obj: list of all detected moving objects
                another_tracker: the tracker of the other camera that is the owner of FOV
        '''
        list_pair = []
        list_most_familiar = []
        if len(self.un_assigned_detects) > 0:
            list_unassigned_obj = []
            for index in self.un_assigned_detects:
                list_unassigned_obj.append(list_moving_obj[index])
            list_un_pair = [];
            for another in another_tracker.object_in_fov:
                isPair = False
                for obj in self.object_in_fov:
                    if obj.label == another.label:
                        isPair = True
                        break
                if isPair == False:
                    list_un_pair.append(another)


            #list_pair, list_most_familiar = stable_matching(list_unassigned_obj, another_tracker.object_in_fov)
            list_pair, list_most_familiar = stable_matching(list_unassigned_obj, list_un_pair)

        for i in range(len(self.un_assigned_detects)):
            # in the case that there are no object in another camera move to this camera
            if list_moving_obj[self.un_assigned_detects[i]] not in list_pair:
                track = Track(self.trackIdCount, list_moving_obj[self.un_assigned_detects[i]])
                self.trackIdCount += 1
                self.tracks.append(track)
            else:
                index = list_pair.index(list_moving_obj[self.un_assigned_detects[i]])
                track = Track(list_most_familiar[index].label, list_moving_obj[self.un_assigned_detects[i]])
                self.tracks.append(track)
    def UpdateVersion1(self, list_moving_obj):
        """
        Args:
            list_moving_obj: detected moving objects to be tracked
        Return:
            None
        """
        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(list_moving_obj)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(list_moving_obj)):
                try:
                    previousObj = self.tracks[i].moving_obj
                    bb_test = [previousObj.topLeft[0][0], previousObj.topLeft[1][0], previousObj.bottomRight[0][0], previousObj.bottomRight[1][0]]
                    bb_gt = [list_moving_obj[j].topLeft[0][0], list_moving_obj[j].topLeft[1][0], list_moving_obj[j].bottomRight[0][0], list_moving_obj[j].bottomRight[1][0]]
                    iou = iou_compute(bb_test, bb_gt)

                    diff = self.tracks[i].prediction - list_moving_obj[j].center
                    diff_feature = 1 / self.tracks[i].moving_obj.compare_other_without_vector(list_moving_obj[j])
                    distances = np.sqrt(diff[0][0]*diff[0][0] +
                                       diff[1][0]*diff[1][0])
                    distance = 1 - iou

                    cost[i][j] = WEIGHT[0] * distance + WEIGHT[1] * diff_feature
                except:
                    pass

        # Let's average the squared ERROR
        cost = (0.5) * cost
        # Using Hungarian Algorithm assign the correct detected measurements to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        print('# cost')
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                print('id: {0}; cost: {1}'.format(self.tracks[i].track_id, cost[i][assignment[i]]))
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if (cost[i][assignment[i]] > self.dist_thresh):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                move_obj = self.tracks[i].moving_obj
                if move_obj.bounding_box.is_under_of_occlusion:
                    # if the previous frame obj is under occlusion => update moving obj by predict value
                    move_obj.update_bbx()
                    move_obj.get_feature()
                else:
                    self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                else:
                    print("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        self.un_assigned_detects = []
        self.object_in_fov = []
        un_assigned_detects_out_fov = []
        for i in range(len(list_moving_obj)):
            # check if the selected moving object is in the fov
            existed_in_fov = self.fov.check_moving_obj_inside_FOV(list_moving_obj[i])
            # add object in fov to list
            if existed_in_fov:
                self.get_position_in_fov(list_moving_obj[i])
                self.object_in_fov.append(list_moving_obj[i])
            if i not in assignment:
                #un_assigned_detects_out_fov.append(i)
                # if the moving object is unassigned but it's inside FOV => it was assigned in the other camera
                if existed_in_fov:
                    self.un_assigned_detects.append(i)
                else:
                    un_assigned_detects_out_fov.append(i)

        # Start new tracks with the moving object that first appear in camera
        if(len(un_assigned_detects_out_fov) != 0):
            for i in range(len(un_assigned_detects_out_fov)):
                track = Track(self.trackIdCount, list_moving_obj[un_assigned_detects_out_fov[i]])
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct(list_moving_obj[assignment[i]].center, 1)
                list_moving_obj[assignment[i]].set_label(self.tracks[i].track_id)
                self.tracks[i].moving_obj = list_moving_obj[assignment[i]]
            else:
                if self.tracks[i].skipped_frames == 0:
                    # in the case the moving obj is under occlusion => predict by current state
                    self.tracks[i].prediction = self.tracks[i].KF.correct(self.tracks[i].moving_obj.center, 1)
                else:
                    self.tracks[i].prediction = self.tracks[i].KF.correct(np.array([[0], [0]]), 0)
            # update the predict attribute in moving object
            move_obj = self.tracks[i].moving_obj
            px = self.tracks[i].prediction[0][0]
            py = self.tracks[i].prediction[1][0] - move_obj.bounding_box.height
            self.tracks[i].moving_obj.update_predict_bbx(px, py)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) - self.max_trace_length):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction
    def get_position_in_fov(self, moving_obj:MovingObject):
        '''
            Description:
                get the distance from moving object to the edge of fov
            Params:
                moving_obj: moving object
            Returns:
                the nearest distance
        '''
        nearest_point = self.fov.get_nearest_point_from_given_point(moving_obj.bounding_box.center)
        # vector = (int(moving_obj.bounding_box.center.x - nearest_point.x), int(moving_obj.bounding_box.center.y - nearest_point.y))
        if self.fov.is_automatic is True:
            moving_obj.set_vector(nearest_point)
        return nearest_point
    def update_cam1(self, list_moving_obj):
        """
        Args:
            list_moving_obj: detected moving objects to be tracked
        Return:
            None
        """
        N = len(self.tracks)
        M = len(list_moving_obj)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(list_moving_obj)):
                try:
                    previousObj = self.tracks[i].moving_obj
                    bb_test = [previousObj.topLeft[0][0], previousObj.topLeft[1][0], previousObj.bottomRight[0][0], previousObj.bottomRight[1][0]]
                    bb_gt = [list_moving_obj[j].topLeft[0][0], list_moving_obj[j].topLeft[1][0], list_moving_obj[j].bottomRight[0][0], list_moving_obj[j].bottomRight[1][0]]
                    iou = iou_compute(bb_test, bb_gt)

                    diff = self.tracks[i].prediction - list_moving_obj[j].center
                    diff_feature = 1 / self.tracks[i].moving_obj.compare_other_without_vector(list_moving_obj[j])
                    distances = np.sqrt(diff[0][0]*diff[0][0] +
                                       diff[1][0]*diff[1][0])
                    distance = 1 - iou

                    cost[i][j] = WEIGHT[0] * distance + WEIGHT[1] * diff_feature
                except:
                    pass

        # Let's average the squared ERROR
        cost = (0.5) * cost
        # Using Hungarian Algorithm assign the correct detected measurements to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        print('# cost')
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                print('id: {0}; cost: {1}'.format(self.tracks[i].track_id, cost[i][assignment[i]]))
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if (cost[i][assignment[i]] > self.dist_thresh):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                move_obj = self.tracks[i].moving_obj
                if move_obj.bounding_box.is_under_of_occlusion:
                    # if the previous frame obj is under occlusion => update moving obj by predict value
                    move_obj.update_bbx()
                    #move_obj.get_feature()
                else:
                    self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                else:
                    print("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        self.un_assigned_detects = []
        self.object_in_fov = []
        un_assigned_detects_out_fov = []
        for i in range(len(list_moving_obj)):
            # check if the selected moving object is in the fov
            existed_in_fov = False
            # add object in fov to list
            if existed_in_fov:
                self.get_position_in_fov(list_moving_obj[i])
                self.object_in_fov.append(list_moving_obj[i])
            if i not in assignment:
                #un_assigned_detects_out_fov.append(i)
                # if the moving object is unassigned but it's inside FOV => it was assigned in the other camera
                if existed_in_fov:
                    self.un_assigned_detects.append(i)
                else:
                    un_assigned_detects_out_fov.append(i)

        # Start new tracks with the moving object that first appear in camera
        if(len(un_assigned_detects_out_fov) != 0):
            for i in range(len(un_assigned_detects_out_fov)):
                track = Track(self.trackIdCount, list_moving_obj[un_assigned_detects_out_fov[i]])
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct(list_moving_obj[assignment[i]].center, 1)
                list_moving_obj[assignment[i]].set_label(self.tracks[i].track_id)
                self.tracks[i].moving_obj = list_moving_obj[assignment[i]]
            else:
                if self.tracks[i].skipped_frames == 0:
                    # in the case the moving obj is under occlusion => predict by current state
                    self.tracks[i].prediction = self.tracks[i].KF.correct(self.tracks[i].prediction, 0)
                else:
                    self.tracks[i].prediction = self.tracks[i].KF.correct(np.array([[0], [0]]), 0)
            # update the predict attribute in moving object
            move_obj = self.tracks[i].moving_obj
            px = self.tracks[i].prediction[0][0]
            py = self.tracks[i].prediction[1][0] - move_obj.bounding_box.height
            self.tracks[i].moving_obj.update_predict_bbx(px, py)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) - self.max_trace_length):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction
    def update_single_camera(self, list_moving_obj):
        for i in range(len(self.un_assigned_detects)):
            track = Track(self.trackIdCount, list_moving_obj[self.un_assigned_detects[i]])
            self.trackIdCount += 1
            self.tracks.append(track)
    def assign_label_second_camera_first_frame(self, list_moving_obj, another_tracker):
        for i in range(len(list_moving_obj)):
            # check if the selected moving object is in the fov
            existed_in_fov = self.fov.check_moving_obj_inside_FOV(list_moving_obj[i])
            # add object in fov to list
            if existed_in_fov:
                self.get_position_in_fov(list_moving_obj[i])
                self.object_in_fov.append(list_moving_obj[i])
                self.un_assigned_detects.append(i)
        self.Update_un_assign_detect(list_moving_obj, another_tracker)
        another_tracker.trackIdCount += len(list_moving_obj)
    def assign_label_consistent(self, another_tracker):
        '''
            Description:
                assign for unassigned moving object that exist inside the FOV
            Params:
                list_moving_obj: list of all detected moving objects
                another_tracker: the tracker of the other camera that is the owner of FOV
        '''
        list_pair = []
        list_most_familiar = []
        list_pair, list_most_familiar = stable_matching(self.object_in_fov, another_tracker.object_in_fov)

        for i in range(len(list_pair)):
            track1 = Track(self.trackIdCount, list_pair[i])
            self.tracks.append(track1)

            track2 = Track(self.trackIdCount, list_most_familiar[i])
            another_tracker.tracks.append(track2)
            self.trackIdCount += 1

    def find_index_of_object_in_tracker_by_label(self, trackers, label):
        for index,tracker in enumerate(trackers):
            if tracker.track_id == label:
                return index
        return -1
    def update_cam1_version_old(self, list_moving_obj):
        """
        Args:
            list_moving_obj: detected moving objects to be tracked
        Return:
            None
        """
        if len(list_moving_obj) == 0:
            return
        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(list_moving_obj)):
                track = Track(self.trackIdCount, list_moving_obj[i])
                self.trackIdCount += 1
                self.tracks.append(track)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(list_moving_obj)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(list_moving_obj)):
                try:
                    previousObj = self.tracks[i].moving_obj
                    delta = self.tracks[i].prediction - previousObj.center
                    predictTopLeft = previousObj.topLeft + delta
                    predictTopRight = previousObj.topRight + delta
                    predictBottomLeft = previousObj.bottomLeft + delta
                    predictBottomRight = previousObj.bottomRight + delta
                    diff_1 = predictTopLeft - list_moving_obj[j].topLeft
                    diff_2 = predictTopRight - list_moving_obj[j].topRight
                    diff_3 = predictBottomLeft - list_moving_obj[j].bottomLeft
                    diff_4 = predictBottomRight - list_moving_obj[j].bottomRight
                    distance_1 = np.sqrt(diff_1[0][0]*diff_1[0][0] + diff_1[1][0]*diff_1[1][0])
                    distance_2 = np.sqrt(diff_2[0][0]*diff_2[0][0] + diff_2[1][0]*diff_2[1][0])
                    distance_3 = np.sqrt(diff_3[0][0]*diff_3[0][0] + diff_3[1][0]*diff_3[1][0])
                    distance_4 = np.sqrt(diff_4[0][0]*diff_4[0][0] + diff_4[1][0]*diff_4[1][0])
                    bb_test = [previousObj.topLeft[0][0], previousObj.topLeft[1][0], previousObj.bottomRight[0][0], previousObj.bottomRight[1][0]]
                    bb_gt = [list_moving_obj[j].topLeft[0][0], list_moving_obj[j].topLeft[1][0], list_moving_obj[j].bottomRight[0][0], list_moving_obj[j].bottomRight[1][0]]
                    iou = iou_compute(bb_test, bb_gt)

                    diff = self.tracks[i].prediction - list_moving_obj[j].center
                    diff_feature = 1 / self.tracks[i].moving_obj.compare_other_without_vector(list_moving_obj[j])
                    distances = np.sqrt(diff[0][0]*diff[0][0] +
                                       diff[1][0]*diff[1][0])
                    distance = 1 - iou #distance_1 + distance_2 + distance_3 + distance_4

                    cost[i][j] = WEIGHT[0] * distance + WEIGHT[1] * diff_feature
                except:
                    pass

        # Let's average the squared ERROR
        cost = (0.5) * cost
        # Using Hungarian Algorithm assign the correct detected measurements to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        print('# cost')
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                print('id: {0}; cost: {1}'.format(self.tracks[i].track_id, cost[i][assignment[i]]))
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if (cost[i][assignment[i]] > self.dist_thresh):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                else:
                    print("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        self.un_assigned_detects = []
        self.object_in_fov = []
        un_assigned_detects_out_fov = []
        for i in range(len(list_moving_obj)):
            if i not in assignment:
                un_assigned_detects_out_fov.append(i)

        # Start new tracks with the moving object that first appear in camera
        if(len(un_assigned_detects_out_fov) != 0):
            for i in range(len(un_assigned_detects_out_fov)):
                track = Track(self.trackIdCount, list_moving_obj[un_assigned_detects_out_fov[i]])
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct(list_moving_obj[assignment[i]].center, 1)
                list_moving_obj[assignment[i]].set_label(self.tracks[i].track_id)
                self.tracks[i].moving_obj = list_moving_obj[assignment[i]]
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(np.array([[0], [0]]), 0)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) - self.max_trace_length):
                    del self.tracks[i].trace[j]


            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction

    def update_cam1_new_kalman(self, list_moving_obj):
        """
        Args:
            list_moving_obj: detected moving objects to be tracked
        Return:
            None
        """
        
        N = len(self.tracks)
        M = len(list_moving_obj)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(list_moving_obj)):
                try:
                    previousObj = self.tracks[i].moving_obj
                    bb_test = [previousObj.topLeft[0][0], previousObj.topLeft[1][0], previousObj.bottomRight[0][0], previousObj.bottomRight[1][0]]
                    bb_gt = [list_moving_obj[j].topLeft[0][0], list_moving_obj[j].topLeft[1][0], list_moving_obj[j].bottomRight[0][0], list_moving_obj[j].bottomRight[1][0]]
                    iou = iou_compute(bb_test, bb_gt)
                    distance = 1 - iou

                    cost[i][j] = WEIGHT[0] * distance
                except:
                    pass

        # Let's average the squared ERROR
        cost = (0.5) * cost
        # Using Hungarian Algorithm assign the correct detected measurements to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        print('# cost')
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                print('id: {0}; cost: {1}'.format(self.tracks[i].track_id, cost[i][assignment[i]]))
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if (cost[i][assignment[i]] > self.dist_thresh):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                move_obj = self.tracks[i].moving_obj
                if len(self.tracks[i].trace) > 2 and (move_obj.bounding_box.is_under_of_occlusion or check_obj_disappear(move_obj.bounding_box, move_obj.img_full)):
                    # if the previous frame obj is under occlusion => update moving obj by predict value
                    move_obj.update_bbx()
                    self.tracks[i].under_occlusion_frame += 1
                    #move_obj.get_feature()
                else:
                    self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
            if (self.tracks[i].under_occlusion_frame > self.max_frames_to_skip):
                self.tracks[i].moving_obj.bounding_box.is_under_of_occlusion = False
                self.tracks[i].skipped_frames = 1
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                else:
                    print("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        self.un_assigned_detects = []
        self.object_in_fov = []
        un_assigned_detects_out_fov = []
        for i in range(len(list_moving_obj)):
            # check if the selected moving object is in the fov
            existed_in_fov = False
            # add object in fov to list
            if existed_in_fov:
                self.get_position_in_fov(list_moving_obj[i])
                self.object_in_fov.append(list_moving_obj[i])
            if i not in assignment:
                #un_assigned_detects_out_fov.append(i)
                # if the moving object is unassigned but it's inside FOV => it was assigned in the other camera
                if existed_in_fov:
                    self.un_assigned_detects.append(i)
                else:
                    un_assigned_detects_out_fov.append(i)

        # Start new tracks with the moving object that first appear in camera
        if(len(un_assigned_detects_out_fov) != 0):
            for i in range(len(un_assigned_detects_out_fov)):
                track = Track(self.trackIdCount, list_moving_obj[un_assigned_detects_out_fov[i]])
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].KF.update(list_moving_obj[assignment[i]].bounding_box)
                list_moving_obj[assignment[i]].set_label(self.tracks[i].track_id)
                self.tracks[i].moving_obj = list_moving_obj[assignment[i]]
            else:
                self.tracks[i].KF.kf.P = np.zeros(self.tracks[i].KF.kf.P.shape)
                if self.tracks[i].skipped_frames == 0:
                    # in the case the moving obj is under occlusion => predict by current state
                    #if len(self.tracks[i].trace) > 0:
                    #    new_bbx_predict = self.tracks[i].KF.get_state()
                    #    old_bbx_predict = self.tracks[i].trace[-1]
                    #    dx = new_bbx_predict.pX - old_bbx_predict.pX
                    #    dy = new_bbx_predict.pY - old_bbx_predict.pY
                    #    dx = int(dx/max(abs(dx), 1))
                    #    dy = int(dy/max(abs(dy), 1))
                    #    self.tracks[i].moving_obj.bounding_box.pX += dx
                    #    self.tracks[i].moving_obj.bounding_box.pY += dy
                    self.tracks[i].KF.update(self.tracks[i].moving_obj.bounding_box)
                else:
                    self.tracks[i].KF.update(self.tracks[i].moving_obj.bounding_box)
            # update the predict attribute in moving object
            move_obj = self.tracks[i].moving_obj
            new_bbx = self.tracks[i].KF.get_state()
            self.tracks[i].moving_obj.update_predict_bbx(new_bbx.pX, new_bbx.pY)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) - self.max_trace_length):
                    del self.tracks[i].trace[j]
            if self.tracks[i].under_occlusion_frame == 0:
                self.tracks[i].trace.append(new_bbx)