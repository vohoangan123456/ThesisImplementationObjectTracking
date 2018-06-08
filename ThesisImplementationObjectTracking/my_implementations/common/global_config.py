AUTO_FOV_COMPUTE = True
THRESHOLD_Y = 20
LIST_FEATURE_EXTRACTION = ['momentColor', 'huInvariance', 'colorHistogram', 'sift', 'surf']
WEIGHTS = [30000, 50, 1, 1, 1.25] # weight of the features that use for compute different between two object
THRESHOLD_ACCEPT = 200
THRESHOLD_INSIDE_OBJ = 20
THRESHOLD_CONFIDENCE = 0.5
THRESHOLD_SIZE = 7000
THRESHOLD = 160
WEIGHT = [100,1]  # [diff_distance, diff_feature]
FOV_OF_CAM2_IN_CAM1 = './fov_computing/cam2Incam1.png'
FOV_OF_CAM1_IN_CAM2 = './fov_computing/cam1Incam2.png'
THRESHOLD_SIZE_CHANGE = 1.5 #150%