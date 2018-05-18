import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
import my_working_space.get_points as get_points
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString
from itertools import product

class Utils:
    def convert_homography_to_polygon(self, homography_point):
        width,heigh,deep = homography_point.shape
        list_point = []
        for x in range(0, width):
            for y in range(0, heigh):
                xy = []
                for z in range(0, deep):
                    xy.append(homography_point[x][y][z])
                list_point.append((xy[0],xy[1]))
        return list_point

class CommonFOV:
    def __init__(self, list_point, camera = 1):
        self.list_point:[] = list_point
        self.polygon:Polygon = Polygon(self.list_point)
        self.camera_name:int = camera

    def check_point_inside_FOV(self, point:Point):
        return self.polygon.contains(point)

    def get_distance_to_edge(self, point:Point):
        return self.polygon.exterior.distance(point)

class BoundingBox:
    def __init__(self, pX, pY, width, height):
        self.pX = pX
        self.pY = pY
        self.width = width
        self.height = height

class MovingObject:
    def __init__(self):
        self.bounding_box = None
        self.image = None
        self.crop_object_img = None

    def create_object_with_boundingbox(self, image, bounding_box:BoundingBox):
        self.bounding_box = bounding_box
        self.image = image
        self.crop_object_img = image[
                               self.bounding_box.pY:self.bounding_box.pY + self.bounding_box.height,
                               self.bounding_box.pX:self.bounding_box.pX + self.bounding_box.width
                               ]

def get_points_process(img):
    # Co-ordinates of objects to be tracked
    # will be stored in a list named `points`
    points = get_points.run(img)
    return points

def crop_img_process(img):
    points = get_points_process(img)
    if not points:
        print("ERROR: No object to be tracked.")
        exit()
    pX = points[0][0]
    pY = points[0][1]
    width = abs(points[0][2] - pX)
    height = abs(points[0][3] - pY)
    bounding_box = BoundingBox(pX, pY, width, height)
    moving_object = MovingObject()
    moving_object.create_object_with_boundingbox(img, bounding_box)
    return moving_object.crop_object_img

def find_fov():
    utils = Utils()
    file_path = './sample_img/'
    img1 = cv2.imread(file_path + 'cam1.jpg', 0)
    img1 = imutils.resize(img1, width=500)
    img2 = cv2.imread(file_path + 'cam2.jpg', 0)
    img2 = imutils.resize(img2, width=500)

    #img1 = crop_img_process(img1)
    sift = cv2.xfeatures2d.SURF_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good) > 10:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        x, y = [], []
        for element in np.int32(dst):
            for item in element:
                x.append(item[0])
                y.append(item[1])
        x1, x2, y1, y2 = min(x), max(x), min(y), max(y)
        #img2 = cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #img2 = cv2.polylines(img2,np.int32(dst),True,255,3, cv2.LINE_AA)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, (255, 255, 255), 2, cv2.LINE_AA)

    else:
        print ("Not enough matches are found - %d/%d" % (len(good),10))
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    cv2.imshow('gray', img3)
    cv2.waitKey(0)

def draw_point(fov, img):
    points = get_points_process(img)
    if not points:
        print("ERROR: No object to be tracked.")
        exit()
    pX = points[0][0]
    pY = points[0][1]

    nearest = fov.polygon.exterior.project(Point(pX, pY))
    point_near = fov.polygon.exterior.interpolate(nearest)
    vector = (int(point_near.x - pX), int(point_near.y - pY))
    print(point_near, point_near.x, point_near.y)
    print('distance: ',fov.get_distance_to_edge(Point(pX, pY)))
    print('Vector: ', str(vector[0]) +'x +' + str(vector[1]) + 'y = 0')
    cv2.line(img, (int(pX), int(pY)), (int(point_near.x), int(point_near.y)), (0, 255, 0), 2)
    return vector


def draw_polygon(polygon:Polygon, img):
    boundary = polygon.boundary.xy
    for index in range(0, len(boundary[0])-1):
        cv2.line(img, (int(boundary[0][index]),int(boundary[1][index])), 
                    (int(boundary[0][index + 1]), int(boundary[1][index + 1])), (0, 0, 0), 6)
def fov_handle():
    from shapely import wkt
    utils = Utils()
    file_path = './sample_img/'
    img1 = cv2.imread(file_path + 'cam1_intown.jpg', 0)
    img1 = imutils.resize(img1, width=500)
    img2 = cv2.imread(file_path + 'cam2_intown.jpg', 0)
    img2 = imutils.resize(img2, width=500)

    #img1 = crop_img_process(img1)
    sift = cv2.xfeatures2d.SURF_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # create FOV img1 in img2
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good) > 10:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        list_point = utils.convert_homography_to_polygon(np.int32(dst))
        fov = CommonFOV(list_point)
        #img2 = cv2.polylines(img2, [np.int32(dst)], True, (255, 255, 255), 2, cv2.LINE_AA)

        cam1_polygon = Polygon([(0,0),(0, h), (w, h), (w, 0)])
        cv2.waitKey(0)

    # create FOV img2 in img1
    matches1 = flann.knnMatch(des2,des1,k=2)
    good1 = []
    for m,n in matches1:
        if m.distance < 0.7*n.distance:
            good1.append(m)
    if len(good1) > 10:
        src_pts1 = np.float32([ kp2[m.queryIdx].pt for m in good1 ]).reshape(-1,1,2)
        dst_pts1 = np.float32([ kp1[m.trainIdx].pt for m in good1 ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts1, dst_pts1, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img2.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        list_point = utils.convert_homography_to_polygon(np.int32(dst))
        fov1 = CommonFOV(list_point)
        #img1 = cv2.polylines(img1, [np.int32(dst)], True, (255, 255, 255), 2, cv2.LINE_AA)

        cam2_polygon = Polygon([(0,0),(0, h), (w, h), (w, 0)])
        fov.polygon = fov.polygon.intersection(cam2_polygon)
        draw_polygon(fov.polygon, img2)

        #fov1.polygon = fov1.polygon.intersection(cam2_polygon)
        draw_polygon(fov1.polygon, img1)

        cv2.waitKey(0)        
    while True:
        cv2.imshow('img2',img2)
        cv2.imshow('img1',img1)
        if cv2.waitKey(18) == ord('e'):
            draw_point(fov, img2)
            draw_point(fov1, img1)
        
def run_two_camera():
    cam1 = cv2.VideoCapture('./videos/sample_video/campus7-c0.avi')
    cam2 = cv2.VideoCapture('./videos/sample_video/campus7-c1.avi')
    cam3 = cv2.VideoCapture('./videos/sample_video/campus7-c2.avi')
    #cam4 = cv2.VideoCapture('./videos/sample_video/4p-c3.avi')
    count = 15
    while True:
        (ret, img1) = cam1.read()
        (ret, img2) = cam2.read()
        (ret, img3) = cam3.read()
        #(ret, img4) = cam4.read()

        cv2.imshow('cam1', img1)
        cv2.imshow('cam2', img2)
        cv2.imshow('cam3', img3)
        #cv2.imshow('cam4', img4)
        if count == 15:
            cv2.imwrite("./sample_img/cut_images/%d_cam1.jpg" % count, img1)
            cv2.imwrite("./sample_img/cut_images/%d_cam2.jpg" % count, img2)
            cv2.imwrite("./sample_img/cut_images/%d_cam3.jpg" % count, img3)
            #cv2.imwrite("./sample_img/cut_images/%d_cam4.jpg" % count, img4)
            print('cut_image_%d', count)
            count += 1
        if cv2.waitKey(18) == ord('e'):
            # save frame as JPEG file
            cv2.imwrite("./sample_img/cut_images/%d_cam1.jpg" % count, img1)
            cv2.imwrite("./sample_img/cut_images/%d_cam2.jpg" % count, img2)
            cv2.imwrite("./sample_img/cut_images/%d_cam3.jpg" % count, img3)
            print('cut_image_%d', count)
            count += 1
        if cv2.waitKey(1) == 27:
            break

def crop_video(video_path, out_path):
    cam1 = cv2.VideoCapture(video_path)
    frame_width = int(cam1.get(3) * 2 / 3) 
    frame_height = int(cam1.get(4))
    out = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    save = False
    while True:
        (ret, img1) = cam1.read()
        if img1 is None:
            break
        key = cv2.waitKey(1)
        if key == ord('e'):
            save = True
        cv2.imshow('cam1', img1)
        if save is True:
            out.write(img1)

        if key == ord('q'):
            break
    cam1.release()
    out.release()
    cv2.destroyAllWindows()
def devide_video(video_path, out_path1, out_path2):
    cam1 = cv2.VideoCapture(video_path)
    one_part = int(cam1.get(3) / 3)
    frame_width = one_part * 2 
    frame_height = int(cam1.get(4))
    out1 = cv2.VideoWriter('./videos/my_video/cam1_left_video1_part_{0}.avi'.format(5),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    out2 = cv2.VideoWriter('./videos/my_video/cam1_left_video2_part_{0}.avi'.format(5),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    save = False
    cut_index = 0
    devideIndex = 1
    while True:
        (ret, img1) = cam1.read()
        if img1 is None or cut_index == 6000:
            break
        output_img1 = img1[
                            0:frame_height,
                            0:frame_width
                        ]
        output_img2 = img1[
                            0:frame_height,
                            one_part: one_part + frame_width
                        ]
        
        key = cv2.waitKey(1)
        if key == ord('e'):
            save = True
        elif key == ord('r'):
            save = False
        elif key == ord('q'):
            break
        elif key == ord('n'):
            devideIndex += 1
            cut_index = 0
            out1 = cv2.VideoWriter('./videos/my_video/cam1_left_video1_part_{0}.avi'.format(devideIndex),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
            out2 = cv2.VideoWriter('./videos/my_video/cam1_left_video2_part_{0}.avi'.format(devideIndex),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
        cv2.putText(output_img1, str('frame: {0}'.format(cut_index)), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('cam1', output_img1)
        cv2.putText(output_img2, str('frame: {0}'.format(cut_index)), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('cam2', output_img2)
        if save is True:
            cut_index += 1
            out1.write(output_img1)
            out2.write(output_img2)
    cam1.release()
    out1.release()
    out2.release()
    cv2.destroyAllWindows()

def play_multiple_video(video_path1, video_path2):
    cam1 = cv2.VideoCapture(video_path1)
    cam2 = cv2.VideoCapture(video_path2)
    time = 18
    pause = False
    while True:
        (ret, img1) = cam1.read()
        (ret, img2) = cam2.read()
        img1 = cv2.resize(img1, (600,600))
        img2 = cv2.resize(img2, (600,600))

        cv2.imshow('cam1', img1)
        cv2.imshow('cam2', img2)
        key = cv2.waitKey(time)
        if key == ord('r'):
            time -= 2
        elif key == ord('e'):
            time += 2
        if key == ord('p'):
            pause = not pause
        while pause:
            key1 = cv2.waitKey(time)
            if key1 == ord('p'):
                pause = not pause

        if key == 27:
            break

def merge_two_video(video_path1, video_path2):
    import imageio
    imageio.plugins.ffmpeg.download()
    from moviepy.editor import VideoFileClip, clips_array
    from random import randint
    rd_number = randint(0, 100) # random number for save video
    clip1 = VideoFileClip(video_path1)
    clip2 = VideoFileClip(video_path2)
    #final_clip = concatenate_videoclips([clip1,clip2])
    #final_clip.write_videofile('./videos/merge_video.mp4')
    final_clip = clips_array([[clip1, clip2]])
    final_clip.resize(width=1200).write_videofile("./videos/merge_file_width_campusc4_{0}.mp4".format(str(rd_number)))
    print('done')

    #rd_number = randint(0, 100) # random number for save video
    #cam1 = cv2.VideoCapture(video_path1)
    #cam2 = cv2.VideoCapture(video_path2)
    #frame_width = int(cam1.get(3)) + int(cam2.get(3))
    #frame_height = int(cam1.get(4))
    #out1 = cv2.VideoWriter('./videos/merge_{0}'.format(str(rd_number)),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    #while True:
    #    (ret, img1) = cam1.read()
    #    (ret, img2) = cam2.read()
    #    img1 = cv2.resize(img1, (600,600))
    #    img2 = cv2.resize(img2, (600,600))
    #    # merge two images
    #    result = Image.new("RGB", (1200, 600))
    #    files = [img1, img2]
    #    for index, img in enumerate(files):
    #        path = os.path.expanduser('./sample_img/cam1.jpg')
    #        img = Image.open(path)
    #        img.thumbnail((400, 400), Image.ANTIALIAS)
    #        x = index // 2 * 400
    #        y = index % 2 * 400
    #        w, h = img.size
    #        result.paste(img, (x, y, x + w, y + h))
    #    #total_width = 1200
    #    #max_height = 600
    #    #new_im = Image.new('RGB', (total_width, max_height))
    #    #new_im.paste(img1,(0,0,600,600))
    #    #new_im.paste(img1, (0, 0, 600, 600))
    #    #new_im.paste(img2, (600,0, 1200, 600))
    #    # end merge

    #    cv2.imshow('cam1', img1)
    #    cv2.imshow('cam2', img2)
    #    cv2.imshow('merge', new_im)
    #    out1.write(new_im)
    #    key = cv2.waitKey(time)

    #    if key == 27:
    #        break
    #cam1.release()
    #cam2.release()
    #out1.release()
    #cv2.destroyAllWindows()

def stable(rankings, A, B):
    partners = dict((a, (rankings[(a, 1)], 1)) for a in A)
    is_stable = False # whether the current pairing (given by `partners`) is stable
    while is_stable == False:
        is_stable = True
        for b in B:
            is_paired = False # whether b has a pair which b ranks <= to n
            for n in range(1, len(B) + 1):
                a = rankings[(b, n)]
                a_partner, a_n = partners[a]
                if a_partner == b:
                    if is_paired:
                        is_stable = False
                        partners[a] = (rankings[(a, a_n + 1)], a_n + 1)
                    else:
                        is_paired = True
    return sorted((a, b) for (a, (b, n)) in partners.items())

def run_with_previous(videopath):
    import copy
    cam1 = cv2.VideoCapture(videopath)
    previous = None
    while True:
        (ret, img1) = cam1.read()
        img1 = cv2.resize(img1, (600,600))

        cv2.imshow('cam1', img1)
        if previous is not None:
            cv2.imshow('previous', previous)
        key = cv2.waitKey(50)
        if key == 27:
            break
        previous = copy.copy(img1)

def save_video(videopath, outputPath):
    frame = 0;
    cut_index = 0;
    cam1 = cv2.VideoCapture(videopath)
    cam2 = cv2.VideoCapture('./videos/my_video/cam2_2.mp4')
    cam3 = cv2.VideoCapture('./videos/my_video/cam2_3.mp4')
    cam4 = cv2.VideoCapture('./videos/my_video/cam2_4.mp4')
    cam5 = cv2.VideoCapture('./videos/my_video/cam2_5.mp4')
    frame_width = int(cam1.get(3))
    frame_height = int(cam1.get(4))
    out1 = cv2.VideoWriter('./videos/my_video/cam1_left_part_{0}.avi'.format(1),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    save = False
    camIndex = 1
    while True:
        frame += 1
        (ret, img1) = cam1.read()
        if img1 is None:
            break
            if camIndex == 1:
                cam1 = cam2
            elif camIndex == 2:
                break
                cam1 = cam3
            elif camIndex == 3:
                cam1 = cam4
            elif camIndex ==4:
                cam1 = cam5
            elif camIndex == 5:
                break
            camIndex += 1
            (ret, img1) = cam1.read()
        if cv2.waitKey(1) == ord('e') or frame == 607:
            if save == False:
                frame = 1
            save = True
        cv2.putText(img1, str('frame: {0}'.format(cut_index)), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('cam1', img1)
        if save is True:
            cut_index += 1
            out1.write(img1)
        if frame >= 3500 and frame % 3500 == 0:
            out1.release()
            out1 = cv2.VideoWriter('./videos/my_video/cam1_left_part_{0}.avi'.format(int(frame/3500 + 1)),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
        if cv2.waitKey(1) == ord('q'):
            break
    cam1.release()
    out1.release()
    cv2.destroyAllWindows()
def crop_first_image(videopath, outpath):
    cam1 = cv2.VideoCapture(videopath)
    (ret, img1) = cam1.read()
    cv2.imwrite(outpath, img1)
    cam1.release()
    cv2.destroyAllWindows()

def get_fov_polygon_from_image(imagePath):
    from my_working_space.kalman_filter.field_of_view import CommonFOV
    from my_working_space.kalman_filter.common import convert_homography_to_polygon
    from shapely.geometry.polygon import Polygon
    img = cv2.imread(imagePath)
    greenLower = (10, 200, 100)
    greenUpper = (64, 255, 255)
    if img is not None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # define range of white color in HSV
        # change it according to your need !
        lower_white = np.array([0,0,200], dtype=np.uint8)
        upper_white = np.array([0,0,255], dtype=np.uint8)

        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(hsv, lower_white, upper_white)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	        cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        # only proceed if at least one contour was found
        if len(cnts) > 0:
	        c = max(cnts, key=cv2.contourArea)
	        list_point = convert_homography_to_polygon(c)
	        fov = CommonFOV()
	        fov.polygon = Polygon(list_point)            
	        fov.draw_polygon(img)
        cv2.imshow('output', img)
        cv2.imshow('mask', mask)
        cv2.waitKey(18);

def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy
def rectangleIntersection():
    from collections import namedtuple
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    # intersection here is (3, 3, 4, 3.5), or an area of 1*.5=.5
    ra = Rectangle(3., 3., 5., 5.)
    rb = Rectangle(4., 1., 6., 3.5)
    print(area(ra, rb))

def read_txt_file(fileName):
    file = open(fileName, "r")
    #lines = file.readlines()
    #print(lines)
    for line in file:
        line = line[:-1].split(', ')
        frame_index = line[0]
        confidence = line[1]
        tl = (line[2], line[3])
        br = (line[4], line[5])
        result = {
            "frame": frame_index,
            "confidence": confidence,
            "topleft": tl,
            "bottomright": br
            }
        print(result)
#run_with_orb();
#find_fov();
#fov_handle();
#draw_polygon();
#run_with_sift();
#run_two_camera()
#crop_video('./videos/my_video/cam2_1.mp4', './videos/my_video/cam2_1_crop_part_1.avi')
devide_video('./videos/my_video/cam1_left.MOV', './videos/my_video/cam1_right_video1.avi', './videos/my_video/cam1_right_video2.avi')
#play_multiple_video('./videos/campus4-c2.avi','./videos/outpy_5_devide_video2_video2.avi')
#merge_two_video('./videos/outpy_93_cam1_right_video1_part_3.avi','./videos/outpy_93_cam1_right_video2_part_3.avi')
#run_with_previous('./videos/outpy_7_videofile_intown.avi')
#save_video('./videos/my_video/cam1_left.MOV', './videos/my_video/cam1_left_part_1.avi')
#crop_first_image('./videos/sample_video/campus4-c1.avi', './sample_img/cut_images/background_2.jpg')
#merge_two_video('./videos/my_video/cam1_left.MOV','./videos/my_video/cam1_right.MOV')
#get_fov_polygon_from_image('./fov_computing/test1.png')
#rectangleIntersection()
#read_txt_file("./videos/detection_code4_cam1.txt")