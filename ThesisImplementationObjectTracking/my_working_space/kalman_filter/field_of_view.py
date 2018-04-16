from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from my_working_space.kalman_filter.common import convert_homography_to_polygon
import numpy as np

class CommonFOV:
    def __init__(self, source_cam_id, target_cam_id):
        '''
            Description:
                create a FOV polygon of target_cam_id inside source_cam_id
            Params:
                source_cam_id: the definition of source camera
                target_cam_id: the definition of target camera
        '''
        self.list_point = []
        self.polygon = None
        self.source_cam_id = source_cam_id
        self.target_cam_id = target_cam_id

    def check_point_inside_FOV(self, point:Point):
        '''
            Description:
                check the point inside or outside of fov
            Params:
                the point need to check
            Return:
                is inside or not
        '''
        if self.polygon is None:
            return False
        return self.polygon.contains(point)

    def get_FOV_of_cam1_in_cam2(self, source_cam, target_cam):
        '''
            Description:
                find the FOV of the target_cam in source_cam
            Params:
                cam1: background image of camera 1
                cam2: background image of camera 2
            return: 
                None
        '''
        # create sift extractor
        sift = cv2.xfeatures2d.SIFT_create()

        # compute keypoint and descriptor of each camera
        kp1, des1 = sift.detectAndCompute(target_cam,None)
        kp2, des2 = sift.detectAndCompute(source_cam,None)

        # create flann matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # find good matches
        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        # find homography of target_cam in source_cam
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = target_cam.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        # create polygon from list points
        self.list_point = convert_homography_to_polygon(np.int32(dst))
        self.polygon = Polygon(self.list_point)