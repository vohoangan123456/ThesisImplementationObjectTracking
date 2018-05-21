'''
    File name         : FOV.py
    File Description  : handle the common field of view (FOV) between two camera
    Author            : An Vo
    Date created      : 19/04/2018
    Python Version    : 3.6
'''
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from my_implementations.common.common import convert_homography_to_polygon, np, cv2
from my_implementations.common.line import Line
from my_implementations.common.global_config import *

class FOV:
    def __init__(self):
        '''
            Description:
                create a FOV polygon of a camera inside another camera
        '''
        self.list_point = []
        self.polygon = None

    def draw_polygon(self, img):
        '''
            Description:
                draw the polygon into img
            Params:
                img: frame image that we need to draw FOV into it
        '''
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

    def check_moving_obj_inside_FOV(self, moving_obj):
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
        return_value = self.polygon.contains(top_left) or self.polygon.contains(bottom_left) or self.polygon.contains(bottom_right) or self.polygon.contains(top_right)
        if return_value is True:
            moving_obj.is_in_fov = True
        return return_value

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

    def get_nearest_point_from_given_point(self, point):
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

    def generate_fov_from_list_point(self, list_point):
        '''
            Description:
                Create fov base on the list point (the vertex of FOV)
            Params:
                list_point([]): the list of vertex of FOV
        '''
        self.list_point = list_point
        if AUTO_FOV_COMPUTE is False:
            list_first = [i[0] for i in list_point]
            minX = min(list_first)
            maxX = max(list_first)
            self.list_point = [i for i in list_point if i[0] == minX or i[0] == maxX]  # just get 4 point of each vertex
            # compute the Line of each edge base on 4 vertex (the order of line is the order i draw in report)
            self.AB = Line(self.list_point[0], self.list_point[1])
            self.BC = Line(self.list_point[1], self.list_point[2])
            self.CD = Line(self.list_point[2], self.list_point[3])
            self.DA = Line(self.list_point[3], self.list_point[0])
        self.polygon = Polygon(self.list_point)