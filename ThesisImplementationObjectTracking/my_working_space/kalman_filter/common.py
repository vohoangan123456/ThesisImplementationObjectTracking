'''
    File name         : common.py
    File Description  : Common debug functions
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/14/2017
    Python Version    : 2.7
'''
from sys import maxsize
import copy
import cv2
import numpy as np

THRESHOLD_ACCEPT = 200
THRESHOLD_INSIDE_OBJ = 20
def convert_homography_to_polygon(homography_point):
    '''
        Description:
            Convert homography point (list point) get from sift match to array
        Params:
            homography_point: sift matches point
        Returns:
            array points
    '''
    width, heigh, deep = homography_point.shape
    list_point = []
    for x in range(0, width):
        for y in range(0, heigh):
            xy = []
            for z in range(0, deep):
                xy.append(homography_point[x][y][z])
            list_point.append((xy[0],xy[1]))
    return list_point

def stable_matching(list_obj1, list_obj2):
    '''
        Description:
            stable matching or GSA (Gale and Shapely Algorithm) use to find the good couple of list object 1 in list object 2
        Params:
            list_obj1: list moving object 1
            list_obj2: list moving object 2
        Returns:
            List tupple contains pair of object from list1 and list2 [(obj1, obj2),(obj1', obj2'),...]
    '''
    list_objA = copy.copy(list_obj1)
    list_objB = copy.copy(list_obj2)
    list_obj_inpair = []        # list object has it pair
    list_most_familiar = []     # list pair of list_object_inpair
    list_diff_value = []        # list difference value of list_obj_inpair and list_most_familiar

    while len(list_obj_inpair) < len(list_objA) and len(list_most_familiar) < len(list_objB):    # loop to all object in list_objA has its pair and all list_objB has its pair
        # narrow list_objB after loop
        list_objB = set(list_objB) - set(list_most_familiar)

        # loop through all object in list_objA to find it pair
        for object in list_objA:
            # get the most familiar pair and their difference value
            (most_familiar, diff_value) = get_most_familiar(object, list_objB)
            if most_familiar is not None:
                # if most_familiar object is single => most_familiar and object are a pair
                if most_familiar not in list_most_familiar:
                    list_obj_inpair.append(object)
                    list_most_familiar.append(most_familiar)
                    list_diff_value.append(diff_value)
                else:   # if most_familiar has already had a couple
                    # find the couple of most_familiar object
                    index = list_most_familiar.index(most_familiar)

                    # if most_familiar resembled object more than the previous couple
                    if list_diff_value[index] > diff_value:
                        # object and most_familiar will become a pair and the previous couple of most_familiar object become single
                        list_obj_inpair[index] = object
                        list_diff_value[index] = diff_value
    return (list_obj_inpair, list_most_familiar)

def get_most_familiar(target, list_object):
    '''
        Description:
            get the most familiar object in list_object for target
        Params:
            target: the object we need to find the most familiar in list_object
            list_object: list all reference object
        Returns:
            _ the object that has the most familiar with target
            _ the different value of two object
    '''
    min_diff = maxsize
    return_obj = None
    for obj in list_object:
        diff = target.compare_other(obj)
        if diff < min_diff:
            min_diff = diff
            return_obj = obj
    return (return_obj, min_diff)

def get_fov_from_image(imagePath):
    img = cv2.imread(imagePath)
    img = cv2.resize(img, (600,600))
    if img is not None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # define range of white color in HSV
        # change it according to your need !
        lower_white = np.array([0,0,100], dtype=np.uint8)
        upper_white = np.array([0,0,255], dtype=np.uint8)

        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(hsv, lower_white, upper_white)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	        cv2.CHAIN_APPROX_SIMPLE)[-2]
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            c = max(cnts, key = cv2.contourArea)
            list_point = convert_homography_to_polygon(c)
            return list_point
    return None

def iou_compute(bb_test,bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)

def check_obj_disappear(bbx, img):
    '''
        Description:
            check if an object is suddenly disappeared in image
        Params:
            bbx: the bounding box of that object
            img: the image was cut from camera
        Returns:
            bool: if object suddenly disappear or it's out of camera
    '''
    bbx.pX
    bbx.pY
    bbx.pXmax
    bbx.pYmax
    w,h,_ = img.shape
    if bbx.area > (THRESHOLD_ACCEPT * 2) and bbx.pX > THRESHOLD_INSIDE_OBJ and bbx.pY > THRESHOLD_INSIDE_OBJ and bbx.pXmax < (w - THRESHOLD_INSIDE_OBJ) and bbx.pYmax < (h - THRESHOLD_INSIDE_OBJ):

        return True
    else:
        return False

def make_pair_corresponse_edge_two_fov(fov1, fov2):
    '''
        Description:
            create a mapping table which edge in fov2 is appropriated with edge in fov1
        Params:
            fov1: the fov 1
            fov2: the fov 2
        Returns: []
            list of pair objects
    '''
    if fov1.is_automatic is False:
        return [(fov1.AB, fov2.DA), (fov1.BC, fov2.AB), (fov1.CD, fov2.BC), (fov1.DA, fov2.CD)]