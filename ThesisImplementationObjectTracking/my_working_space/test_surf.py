import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
import my_working_space.get_points as get_points
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

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
def fov_handle():
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
        list_point = utils.convert_homography_to_polygon(np.int32(dst))
        fov = CommonFOV(list_point)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.waitKey(0)
    while True:
        cv2.imshow('img2',img2)
        points = get_points_process(img2)
        if not points:
            print("ERROR: No object to be tracked.")
            exit()
        pX = points[0][0]
        pY = points[0][1]
        if fov.check_point_inside_FOV(Point(pX, pY)):
            print('inside')
        else:
            print('outside')
        
def run_two_camera():
    cam1 = cv2.VideoCapture('./videos/4p-c0.avi')
    cam2 = cv2.VideoCapture('./videos/4p-c1.avi')
    cam3 = cv2.VideoCapture('./videos/4p-c2.avi')
    cam4 = cv2.VideoCapture('./videos/4p-c3.avi')
    count = 14
    while True:
        (ret, img1) = cam1.read()
        (ret, img2) = cam2.read()
        (ret, img3) = cam3.read()
        (ret, img4) = cam4.read()

        cv2.imshow('cam1', img1)
        cv2.imshow('cam2', img2)
        cv2.imshow('cam3', img3)
        cv2.imshow('cam4', img4)
        if count == 14:
            cv2.imwrite("./sample_img/cut_images/%d_cam1.jpg" % count, img1)
            cv2.imwrite("./sample_img/cut_images/%d_cam2.jpg" % count, img2)
            cv2.imwrite("./sample_img/cut_images/%d_cam3.jpg" % count, img3)
            cv2.imwrite("./sample_img/cut_images/%d_cam4.jpg" % count, img4)
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
    frame_width = int(cam1.get(3))
    frame_height = int(cam1.get(4))
    out = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    save = False
    while True:
        (ret, img1) = cam1.read()
        if img1 is None:
            break
        if cv2.waitKey(18) == ord('e'):
            save = True
        cv2.imshow('cam1', img1)
        if save is True:
            out.write(img1)

        if cv2.waitKey(18) == ord('q'):
            break
    cam1.release()
    out.release()
    cv2.destroyAllWindows()

#run_with_orb();
#find_fov();
fov_handle();
#run_with_sift();
#run_two_camera()
#crop_video('./videos/videofile_inroom.avi', './videos/video2.avi')