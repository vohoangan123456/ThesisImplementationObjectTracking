from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from my_working_space.kalman_filter.common import convert_homography_to_polygon
from my_working_space.kalman_filter.moving_object import MovingObject
import numpy as np
import cv2

class CommonFOV:
    def __init__(self):
        '''
            Description:
                create a FOV polygon of a camera inside another camera
        '''
        self.list_point = []
        self.polygon = None
    def draw_polygon(self, img):
        boundary = self.polygon.boundary.xy
        for index in range(0, len(boundary[0])-1):
            cv2.line(img, (int(boundary[0][index]),int(boundary[1][index])), 
                        (int(boundary[0][index + 1]), int(boundary[1][index + 1])), (0, 0, 0), 3)

    def check_point_inside_FOV(self, point):
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
        return self.polygon.contains(Point(point[0],point[1]))

    def check_moving_obj_inside_FOV(self, moving_obj:MovingObject):
        '''
            Description: 
                check a moving object is inside a FOV or not
            Params:
                moving_obj: detected moving object
            Returns:
                inside or not
        '''

        # get top_left, bottom_left, bottom_right, top_right vertex of a bouding box
        top_left = Point(moving_obj.bounding_box.pX, moving_obj.bounding_box.pY)
        bottom_left = Point(moving_obj.bounding_box.pX, moving_obj.bounding_box.pY + moving_obj.bounding_box.height)
        bottom_right = Point(moving_obj.bounding_box.pX + moving_obj.bounding_box.width, moving_obj.bounding_box.pY + moving_obj.bounding_box.height)
        top_right = Point(moving_obj.bounding_box.pX + moving_obj.bounding_box.width, moving_obj.bounding_box.pY)   
        
        # return true if any of vertex is inside FOV
        return self.polygon.contains(top_left) or self.polygon.contains(bottom_left) or self.polygon.contains(bottom_right) or self.polygon.contains(top_right)

    def get_FOV_of_target_in_source(self, target_cam, source_cam):
        '''
            Description:
                find the FOV of the target_cam in source_cam
            Params:
                target_cam: background image of camera target
                source_cam: background image of camera source
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

        h,w,_ = target_cam.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        # create polygon from list points
        self.list_point = convert_homography_to_polygon(np.int32(dst))
        self.polygon = Polygon(self.list_point)

        # create polygon of camera
        cam_polygon = Polygon([(0,0),(0, h), (w, h), (w, 0)])

        # get the intersection between the fov and the camera
        self.polygon = self.polygon.intersection(cam_polygon)

    def get_nearest_point_from_given_point(self, point:Point):
        '''
            Description: 
                get the nearest distance in the boundary of polygon from the given point
            Params:
                point: the given point
            Returns:
                the distance that closest with the given point
        '''
        nearest_distance = self.polygon.exterior.project(point)
        nearest_point = self.polygon.exterior.interpolate(nearest_distance)
        return nearest_point