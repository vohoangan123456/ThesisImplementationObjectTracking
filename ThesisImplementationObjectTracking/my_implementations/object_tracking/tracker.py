'''
    File name         : tracker.py
    File Description  : handle the tracker object 
    Author            : An Vo
    Date created      : 19/04/2018
    Python Version    : 3.6
'''
from scipy.optimize import linear_sum_assignment
from my_implementations.common.global_config import *
from my_implementations.common.moving_object import MovingObject
from my_implementations.common.FOV import FOV, np
from my_implementations.common.kalman_filter import KalmanFilterTracker
from my_implementations.common.common import stable_matching, get_fov_from_image, iou_compute, check_obj_disappear

class Track(object):
    def __init__(self, trackId, moving_obj):
        self.moving_obj = moving_obj            # moving object that be tracked
        self.moving_obj.set_label(trackId)
        self.track_id = trackId            # identification of each track object
        self.KF = KalmanFilterTracker(moving_obj.bounding_box)                # KF instance to track this object
        self.skipped_frames = 0                 # number of frames skipped undetected
        self.under_occlusion_frame = 0          # number of frames under occlusion
        self.trace = []                         # trace path

class Tracker(object):
    def __init__(self, threshold, max_skip_frames, max_trace_length, trackId, camId):
        '''
            Description:
                Initialize tracker object
            Params:
                threshold(int): the threshold to check the detected object is new object or already existed
                max_skip_frames(int): the number of frame that allow the tracker keep alive
                max_trace_length(int): the number of history is stored in trace
                trackId(int): the tracker id
                camId(int): camera identifier
        '''
        self.threshold = threshold
        self.max_skip_frames = max_skip_frames
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackId = trackId
        self.un_assigned_detects = []
        self.object_in_fov = []
        self.fov = FOV()
        self.camera_id = camId

    def get_FOV(self, target_cam, source_cam, img_path = None):
        '''
            Description:
                get the fov of target camera in source camera
            Params:
                target_cam(frame): the frame of camera that need to find it's fov in the source camera
                source_cam(frame): the source camera that have the common FOV with target camera
                img_path
        '''
        if AUTO_FOV_COMPUTE is False:
            # the case fov is calculate manually
            list_point = get_fov_from_image(img_path)
            self.fov.generate_fov_from_list_point(list_point)
        else:
            self.fov.get_FOV_of_target_in_source(target_cam, source_cam)

    def set_trackId(self, trackId):
        self.trackId = trackId

    def Update(self, list_moving_obj):
        '''
            Description:
                update tracker state
            Params:
                list_moving_obj([]): the list of detected moving object
        '''
        # Calculate different between the tracker objects and detected moving objects
        N = len(self.tracks)
        M = len(list_moving_obj)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            print('TrackId: ', self.tracks[i].track_id)
            for j in range(len(list_moving_obj)):
                try:
                    previousObj = self.tracks[i].moving_obj
                    bb_test = [previousObj.topLeft[0][0], previousObj.topLeft[1][0], previousObj.bottomRight[0][0], previousObj.bottomRight[1][0]]
                    bb_gt = [list_moving_obj[j].topLeft[0][0], list_moving_obj[j].topLeft[1][0], list_moving_obj[j].bottomRight[0][0], list_moving_obj[j].bottomRight[1][0]]
                    iou = iou_compute(bb_test, bb_gt)
                    distance = 1 - iou

                    pre_bbx = self.tracks[i].KF.get_current_state()
                    distances = np.sqrt((pre_bbx.pX - list_moving_obj[j].bounding_box.pX)**2 + (pre_bbx.pYmax - list_moving_obj[j].bounding_box.pYmax)**2)

                    diff_feature = self.tracks[i].moving_obj.compare_features(list_moving_obj[j])
                    cost[i][j] = (WEIGHT[0] * distance + WEIGHT[1] * distances) * WEIGHTS[3] + diff_feature * WEIGHTS[4]
                    print('cost obj:', cost[i][j])
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
                if (cost[i][assignment[i]] < self.threshold) or (cost[i][assignment[i]] < (self.threshold + 30) and self.tracks[i].moving_obj.bounding_box.is_under_of_occlusion):
                    pass
                else:
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
            else:
                move_obj = self.tracks[i].moving_obj
                self.tracks[i].moving_obj.bounding_box.is_disappear = check_obj_disappear(move_obj.bounding_box, move_obj.img_full)
                if len(self.tracks[i].trace) > 2 and (move_obj.bounding_box.is_under_of_occlusion or move_obj.bounding_box.is_disappear):
                    print('id: {0}; is-disappear: {1}'.format(self.tracks[i].track_id, str(move_obj.bounding_box.is_disappear)))
                    # if the previous frame obj is under occlusion => update moving obj by predict value
                    move_obj.update_bbx()
                    self.tracks[i].under_occlusion_frame += 1
                else:
                    self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_skip_frames):
                del_tracks.append(i)
            if (self.tracks[i].under_occlusion_frame > self.max_skip_frames):
                self.tracks[i].moving_obj.bounding_box.is_under_of_occlusion is False
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
                track = Track(self.trackId, list_moving_obj[un_assigned_detects_out_fov[i]])
                self.trackId += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()
            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].under_occlusion_frame = 0
                self.tracks[i].KF.update(list_moving_obj[assignment[i]].bounding_box)
                list_moving_obj[assignment[i]].set_label(self.tracks[i].track_id)
                totalWidth = 0
                totalHeight = 0
                m_obj = list_moving_obj[assignment[i]]
                for bbx_trace in self.tracks[i].trace:
                    totalWidth += bbx_trace.width
                    totalHeight += bbx_trace.height
                totalWidth = totalWidth / max(len(self.tracks[i].trace), 1)
                totalHeight = totalHeight / max(len(self.tracks[i].trace), 1)
                if totalHeight * totalWidth > m_obj.bounding_box.height * m_obj.bounding_box.width * THRESHOLD_SIZE_CHANGE:
                    m_obj.bounding_box.width = totalWidth
                    m_obj.bounding_box.height = totalHeight
                self.tracks[i].moving_obj = list_moving_obj[assignment[i]]
            else:
                self.tracks[i].KF.kf.P = np.zeros(self.tracks[i].KF.kf.P.shape)
                if self.tracks[i].skipped_frames == 0:
                    # in the case the moving obj is under occlusion => predict by current state
                    if len(self.tracks[i].trace) > 0:
                        new_bbx_predict = self.tracks[i].KF.get_current_state()
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
            new_bbx = self.tracks[i].KF.get_current_state()
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

            list_pair, list_most_familiar = stable_matching(list_unassigned_obj, list_un_pair)

        for i in range(len(self.un_assigned_detects)):
            # in the case that there are no object in another camera move to this camera
            if list_moving_obj[self.un_assigned_detects[i]] not in list_pair:
                track = Track(self.trackId, list_moving_obj[self.un_assigned_detects[i]])
                self.trackId += 1
                self.tracks.append(track)
            else:
                index = list_pair.index(list_moving_obj[self.un_assigned_detects[i]])
                track = Track(list_most_familiar[index].label, list_moving_obj[self.un_assigned_detects[i]])
                self.tracks.append(track)

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
            track1 = Track(self.trackId, list_pair[i])
            self.tracks.append(track1)

            track2 = Track(self.trackId, list_most_familiar[i])
            another_tracker.tracks.append(track2)
            self.trackId += 1

    def get_position_in_fov(self, moving_obj):
        '''
            Description:
                get the distance from moving object to the edge of fov
            Params:
                moving_obj: moving object
            Returns:
                the nearest distance
        '''
        nearest_point = self.fov.get_nearest_point_from_given_point(moving_obj.bounding_box.center)
        if AUTO_FOV_COMPUTE is True:
            moving_obj.set_vector(nearest_point)
        return nearest_point