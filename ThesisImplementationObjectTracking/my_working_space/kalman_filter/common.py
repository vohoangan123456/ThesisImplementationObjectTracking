'''
    File name         : common.py
    File Description  : Common debug functions
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/14/2017
    Python Version    : 2.7
'''

def convert_homography_to_polygon(homography_point):
    width, heigh, deep = homography_point.shape
    list_point = []
    for x in range(0, width):
        for y in range(0, heigh):
            xy = []
            for z in range(0, deep):
                xy.append(homography_point[x][y][z])
            list_point.append((xy[0],xy[1]))
    return list_point