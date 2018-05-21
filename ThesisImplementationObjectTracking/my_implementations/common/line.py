'''
    File name         : line.py
    File Description  : handle the line object 
    Author            : An Vo
    Date created      : 19/04/2018
    Python Version    : 3.6
'''
import numpy as np
class Line:
    def __init__(self, p1, p2):
        '''
            Description
                create Line class
            Params:
                p1(x1, y1): first vertex of line
                p2(x2, y2): second vertex of line
        '''
        self.p1 = p1
        self.p2 = p2
        # line format ax + by + c = 0
        self.a = 0
        self.b = 0
        self.c = 0
        self.generate_line()

    def generate_line(self):
        '''
            Description:
                generate line from two given points
        '''
        self.b = self.p2[0] - self.p1[0]
        self.a = self.p1[1] - self.p2[1]
        self.c = -(self.a*self.p1[0] + self.b*self.p1[1])

    def distance_from_point(self, p):
        '''
            Description:
                Distance from a point p to this line
            Params:
                p(x,y): the point
            Returns: (float)
                - the distance between p to this line
        '''
        # find the right angle line that cross point p with this line
        # the format of right angle line is:ax + by + c = 0 <=> self.bx - self.ay + c = 0
        # find the point that two line are intersected
        # get the distance from point p to that point
        a = self.b
        b = -self.a
        c = -(a * p[0] + b * p[1])
        x = (self.b * c - b * self.c)/ (b * self.a - a * self.b)
        y = (c * self.a - a * self.c)/(a * self.b - self.a * b)
        distance = np.sqrt((p[0] - x)**2 + (p[1] - y)**2)
        return distance