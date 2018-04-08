import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
import my_working_space.get_points as get_points
from my_working_space.Main_Functions import BoundingBox

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
    return BoundingBox(pX, pY, width, height)

def run_with_sift():
    MIN_MATCH_COUNT = 5

    sift = cv2.xfeatures2d.SURF_create()
    #sift = cv2.ORB_create()
    img1 = cv2.imread('./sample_img/video1_img.png',0)          # queryImage
    #img1 = imutils.resize(img1, width=800)
    kp1, des1 = sift.detectAndCompute(img1,None)

    #img2 = cv2.imread('./sample_img/all_object.png',0)          # trainImage

    cam = cv2.VideoCapture('./videos/videofile.mp4')
    while True:
        (ret, img2) = cam.read()
        if cv2.waitKey(18) == ord('e'):
            bbx:BBoundingBox = crop_img_process(img2)
            img1 = img2[bbx.pY:(bbx.pY + bbx.height), bbx.pX:(bbx.pX + bbx.width), 0]
            cv2.imshow('im_out', img1)

        # find the keypoints and descriptors with SIFT
        kp2, des2 = sift.detectAndCompute(img2,None)

        #flann base matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        #matches = bf.match(des1, des2)
        #good = sorted(matches, key = lambda x:x.distance)
        # store all the good matches as per Lowe's ratio test.
        
        

        if len(good) > MIN_MATCH_COUNT:
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
            if x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0:
                img1 = img2[y1:y2, x1:x2, 0]
                cv2.imshow('im_out', img1)
            img2 = cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 1)
            #img2 = cv2.polylines(img2,np.int32(dst),True,255,3, cv2.LINE_AA)
        else:
            print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        cv2.imshow('gray', img3)

        #update the query image
        kp1, des1 = sift.detectAndCompute(img1, None)

        if cv2.waitKey(1) == 27:
            break

def run_with_orb():
    MIN_MATCH_COUNT = 4
    ## Create ORB object and BF object(using HAMMING)
    #orb = cv2.ORB_create()
    orb = cv2.ORB_create()
    img1 = cv2.imread('./sample_img/test1.jpg')
    img1 = imutils.resize(img1, width=400)
    img2 = cv2.imread('./sample_img/test1.1.jpg')
    img2 = imutils.resize(img2, width=400)

    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    ## Find the keypoints and descriptors with ORB
    kpts1, descs1 = orb.detectAndCompute(gray1,None)
    kpts2, descs2 = orb.detectAndCompute(gray2,None)

    ## match descriptors and sort them in the order of their distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descs1, descs2)
    dmatches = sorted(matches, key = lambda x:x.distance)

    ## extract the matched keypoints
    src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
    dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

    ## find homography matrix and do perspective transform
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    h,w = img1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    ## draw found regions
    img2 = cv2.polylines(img2, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)
    cv2.imshow("found", img2)

    ## draw match lines
    res = cv2.drawMatches(img1, kpts1, img2, kpts2, dmatches[:20],None,flags=2)

    cv2.imshow("orb_match", res);

    cv2.waitKey();cv2.destroyAllWindows()

def find_fov():
    img1 = cv2.imread('./sample_img/cam1.jpg', 0)
    img1 = imutils.resize(img1, width=500);
    img2 = cv2.imread('./sample_img/cam2.jpg', 0)
    img2 = imutils.resize(img2, width=500);

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

        img2 = cv2.polylines(img2,np.int32(dst),True,255,3, cv2.LINE_AA)
    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    cv2.imshow('gray', img3)
    cv2.waitKey(0)


#run_with_orb();
find_fov();
#run_with_sift();