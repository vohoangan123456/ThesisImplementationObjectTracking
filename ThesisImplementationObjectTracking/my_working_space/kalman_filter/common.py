'''
    File name         : common.py
    File Description  : Common debug functions
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/14/2017
    Python Version    : 2.7
'''

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

def GSA(list_obj1, list_obj2):
    '''
        Description:
            GSA (Gale and Shapely Algorithm) use to find the good couple of list object 1 in list object 2
        Params:
            list_obj1: list moving object 1
            list_obj2: list moving object 2
        Returns:
            List tupple contains pair of object from list1 and list2 [(obj1, obj2),(obj1', obj2'),...]
    '''
    list_pair = []
    while len(list_pair) < len(list_obj1):
        for object in list_obj1:


def get_most_familiar(target, list_object):
    '''
        Description:
            get the most familiar object in list_object for target
        Params:
            target: the object we need to find the most familiar in list_object
            list_object: list all reference object
        Returns:
            the object that has the most familiar with target
    '''
    for obj in list_object:
        target.