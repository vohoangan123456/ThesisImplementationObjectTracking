import cv2
from darkflow.net.build import TFNet
from imutils import paths
# import matplotlib.pyplot as plt

# %config InlineBackend.figure_format = 'svg'
options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.3,
    'gpu': 1.0
}
tfnet = TFNet(options)
imagePaths = list(paths.list_images("sample_img"))
for imagePath in imagePaths:
    img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # use YOLO to predict the image
    result = tfnet.return_predict(img)

    for item in result:
        tl = (item['topleft']['x'], item['topleft']['y'])
        br = (item['bottomright']['x'], item['bottomright']['y'])
        label = item['label']

        # add the box and label and display it
        img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
        img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('frame', img)
    cv2.waitKey(0)
