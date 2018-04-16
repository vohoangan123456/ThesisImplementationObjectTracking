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
from scipy.optimize import linear_sum_assignment
from my_working_space.kalman_filter.kalman_filter import KalmanFilter
from my_working_space.kalman_filter.moving_object import MovingObject
from my_working_space.kalman_filter.field_of_view import CommonFOV


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
    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length, trackIdCount):
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
        self.fov = CommonFOV()

    def get_FOV(self, target_cam, source_cam):
        # init two FOV of two camera
        self.fov.get_FOV_of_target_in_source(target_cam, source_cam)

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
        self.un_assigned_detects = []
        un_assigned_detects_out_fov = []
        for i in range(len(list_moving_obj)):
                if i not in assignment:
                    # if the moving object is unassigned but it's inside FOV => it was assigned in the other camera
                    if self.fov.check_moving_obj_inside_FOV(list_moving_obj[i]):
                        self.un_assigned_detects.append(i)
                    else:
                        un_assigned_detects_out_fov.append(i)

        # Start new tracks with the moving object that first appear in camera
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

    def Update_un_assign_detect(self, list_moving_obj, another_tracker):
        '''
            Description:
                assign for unassigned moving object that exist inside the FOV
            Params:
                list_moving_obj: list of all detected moving objects
                another_tracker: the tracker of the other camera that is the owner of FOV
        '''
        for i in range(len(self.un_assigned_detects)):
            track = Track(self.trackIdCount, list_moving_obj[self.un_assigned_detects[i]])
            self.trackIdCount += 1
            self.tracks.append(track)