import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
import my_working_space.get_points as get_points
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString

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
    frame_width = int(cam1.get(3) * 2 / 3) 
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
def devide_video(video_path, out_path1, out_path2):
    cam1 = cv2.VideoCapture(video_path)
    one_part = int(cam1.get(3) / 3)
    frame_width = one_part * 2 
    frame_height = int(cam1.get(4))
    out1 = cv2.VideoWriter(out_path1,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    out2 = cv2.VideoWriter(out_path2,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    save = False
    while True:
        (ret, img1) = cam1.read()
        output_img1 = img1[
                            0:frame_height,
                            0:frame_width
                        ]
        output_img2 = img1[
                            0:frame_height,
                            one_part: one_part + frame_width
                        ]
        
        if img1 is None:
            break
        if cv2.waitKey(18) == ord('e'):
            save = True
        cv2.imshow('cam1', output_img1)
        cv2.imshow('cam2', output_img2)
        if save is True:
            out1.write(output_img1)
            out2.write(output_img2)

        if cv2.waitKey(18) == ord('q'):
            break
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
#run_with_orb();
#find_fov();
#fov_handle();
#draw_polygon();
#run_with_sift();
#run_two_camera()
#crop_video('./videos/videofile_intown.avi', './videos/video2.avi')
#devide_video('./videos/videofile_intown.avi', './videos/devide_video1.avi', './videos/devide_video2.avi')
play_multiple_video('./videos/outpy_24_devide_video1.avi','./videos/outpy_24_devide_video2.avi')