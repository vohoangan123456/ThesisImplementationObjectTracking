'''
    File name         : multiTracker.py
    File Description  : Tracker Using Kalman Filter & Hungarian Algorithm
    Author            : An Vo Hoang
    Date created      : 10/02/2018
    Date last modified: 10/02/2018
    Python Version    : 3.6
'''

# Import python libraries
import numpy as np
import sys
import copy
from scipy.optimize import linear_sum_assignment
from my_working_space.kalman_filter.kalman_filter import KalmanFilter
from my_working_space.kalman_filter.moving_object import MovingObject
from my_working_space.kalman_filter.field_of_view import CommonFOV
from my_working_space.kalman_filter.common import stable_matching

THRESHOLD = 160
WEIGHT = [2,1]  # [diff_distance, diff_feature]

class Track(object):
    def __init__(self, trackIdCount, moving_obj):
        self.moving_obj = moving_obj            # moving object that be tracked
        self.moving_obj.set_label(trackIdCount)
        self.track_id = trackIdCount            # identification of each track object
        self.KF = KalmanFilter()                # KF instance to track this object
        self.prediction = np.asarray(moving_obj.center)# predicted centroids (x,y)
        self.skipped_frames = 0                 # number of frames skipped undetected
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
        self.object_out_fov = []
        self.list_moving_obj = []
        self.fov = CommonFOV()
        self.camera_id = camId

    def get_FOV(self, target_cam, source_cam):
        # init two FOV of two camera
        self.fov.get_FOV_of_target_in_source(target_cam, source_cam)

    def set_trackIdCount(self, new_trackIdCount):
        self.trackIdCount = new_trackIdCount

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
        moving_obj.set_vector(nearest_point)
        return nearest_point
    
    def devide_in_out_fov_object(self, list_moving_obj):
        if len(list_moving_obj) == 0:
            return
        self.list_moving_obj = list_moving_obj
        for object in list_moving_obj:
            self.get_position_in_fov(object)

        # Now look for object is out of FOV        
        self.object_in_fov = []
        self.object_out_fov = []
        for i in range(len(list_moving_obj)):
            # check if the selected moving object is in the fov
            existed_in_fov = self.fov.check_moving_obj_inside_FOV(list_moving_obj[i])
            # add object in fov to list
            if existed_in_fov:
                self.object_in_fov.append(list_moving_obj[i])
            else:
                self.object_out_fov.append(list_moving_obj[i])

    def Update_out_fov_obj(self):
        """
        Args:
            list_moving_obj: detected moving objects are out of fov to be tracked
        Return:
            None
        """
        if len(self.list_moving_obj) == 0:
            return        

        # from here, just assign label for object in object_out_fov list
        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(self.object_out_fov)):
                track = Track(self.trackIdCount, self.object_out_fov[i])
                self.trackIdCount += 1
                self.tracks.append(track)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(self.object_out_fov)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(self.object_out_fov)):
                try:
                    diff = self.tracks[i].prediction - self.object_out_fov[j].center
                    diff_vector = self.tracks[i].moving_obj.compare_other(self.object_out_fov[j])
                    distance = np.sqrt(diff[0][0]*diff[0][0] +
                                       diff[1][0]*diff[1][0])

                    cost[i][j] =distance# (WEIGHT[0] * distance / (self.tracks[i].skipped_frames + 1) + WEIGHT[1] * diff_vector)
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
        un_assigned_detects_out_fov = []
        for i in range(len(self.object_out_fov)):
            if i not in assignment:
                # if the moving object is unassigned
                un_assigned_detects_out_fov.append(i)

        # Start new tracks with the moving object that first appear in camera
        if(len(un_assigned_detects_out_fov) != 0):
            for i in range(len(un_assigned_detects_out_fov)):
                track = Track(self.trackIdCount, self.object_out_fov[un_assigned_detects_out_fov[i]])
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct(self.object_out_fov[assignment[i]].center, 1)
                self.object_out_fov[assignment[i]].set_label(self.tracks[i].track_id)
                self.tracks[i].moving_obj = self.object_out_fov[assignment[i]]
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(np.array([[0], [0]]), 0)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) - self.max_trace_length):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction
    def Update(self, list_moving_obj):
        """
        Args:
            list_moving_obj: detected moving objects to be tracked
        Return:
            None
        """

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
                    diff = self.tracks[i].prediction - list_moving_obj[j].center
                    distance = np.sqrt(diff[0][0]*diff[0][0] +
                                       diff[1][0]*diff[1][0])
                    cost[i][j] = distance
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
        for i in range(len(assignment)):
            if (assignment[i] != -1):
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
        un_assigned_detects = []
        for i in range(len(list_moving_obj)):
                if i not in assignment:
                    un_assigned_detects.append(i)
        # Start new tracks
        if(len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                track = Track(self.trackIdCount, list_moving_obj[un_assigned_detects[i]])
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct(list_moving_obj[assignment[i]].center, 1)
                self.tracks[i].moving_obj = list_moving_obj[assignment[i]]
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(np.array([[0], [0]]), 0)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) - self.max_trace_length):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction

class MultiTracker:
    def consistent_label(tracker1, tracker2):
        '''
            Description:
                assign label for object inside fov
            Params:
                list_moving_obj: list of all detected moving objects
                another_tracker: the tracker of the other camera that is the owner of FOV
        '''
        trackIdCount = max(tracker1.trackIdCount, tracker2.trackIdCount)
        list_pair = []
        list_most_familiar = []
        list_pair, list_most_familiar = stable_matching(tracker1.object_in_fov, tracker2.object_in_fov)

        # create new track for object with no matching
        for object in tracker1.object_in_fov:
            if object not in list_pair:
                track1 = Track(trackIdCount, object)
                trackIdCount += 1
                tracker1.tracks.append(track1)

        for object in tracker2.object_in_fov:
            if object not in list_most_familiar:
                track2 = Track(trackIdCount, object)
                trackIdCount += 1
                tracker2.tracks.append(track2)
        # consist label in pair
        for index in range(0, len(list_pair)):
            pair_obj = list_pair[index]
            similar_obj = list_most_familiar[index]

            # if two object hadn't been assigned yet
            if pair_obj.label == 0 and similar_obj.label == 0:
                # create new track for pair object
                track3 = Track(trackIdCount, pair_obj)
                tracker1.tracks.append(track3)
                # create new track for similar object
                track4 = Track(trackIdCount, similar_obj)
                trackIdCount += 1
                tracker2.tracks.append(track4)
            elif pair_obj.label == 0:   # in the case object in cam1 need to assign
                track5 = Track(similar_obj.label, pair_obj)
                tracker1.tracks.append(track5)
            elif similar_obj.label == 0:    # in the case object in cam2 need to assign
                track6 = Track(pair_obj.label, similar_obj)
                tracker2.tracks.append(track6)
            #elif pair_obj.label != similar_obj.label:   # in the case two object are assign different

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(object_out_fov)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(object_out_fov)):
                try:
                    diff = self.tracks[i].prediction - object_out_fov[j].center
                    diff_vector = self.tracks[i].moving_obj.compare_other(object_out_fov[j])
                    distance = np.sqrt(diff[0][0]*diff[0][0] +
                                       diff[1][0]*diff[1][0])

                    cost[i][j] =distance# (WEIGHT[0] * distance / (self.tracks[i].skipped_frames + 1) + WEIGHT[1] * diff_vector)
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
        un_assigned_detects_out_fov = []
        for i in range(len(object_out_fov)):
            if i not in assignment:
                # if the moving object is unassigned
                un_assigned_detects_out_fov.append(i)

        # Start new tracks with the moving object that first appear in camera
        if(len(un_assigned_detects_out_fov) != 0):
            for i in range(len(un_assigned_detects_out_fov)):
                track = Track(self.trackIdCount, object_out_fov[un_assigned_detects_out_fov[i]])
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct(object_out_fov[assignment[i]].center, 1)
                object_out_fov[assignment[i]].set_label(self.tracks[i].track_id)
                self.tracks[i].moving_obj = object_out_fov[assignment[i]]
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(np.array([[0], [0]]), 0)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) - self.max_trace_length):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction
        tracker1.trackIdCount = trackIdCount
        tracker2.trackIdCount = trackIdCount
        
