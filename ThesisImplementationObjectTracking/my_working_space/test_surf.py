import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 5

sift = cv2.xfeatures2d.SIFT_create()
img1 = cv2.imread('./sample_img/video2_img.png',0)          # queryImage
kp1, des1 = sift.detectAndCompute(img1,None)

#img2 = cv2.imread('./sample_img/all_object.png',0)          # trainImage

cam = cv2.VideoCapture('./videos/video1.avi')
while True:
    (ret, img2) = cam.read()

    # find the keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(img2,None)

    #flann base matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

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