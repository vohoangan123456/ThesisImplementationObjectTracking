# import the necessary packages
#from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

def merge_two_dict(dict1, dict2):
    desc1 = dict1['Descriptor']
    desc2 = dict2['Descriptor']
    sumDesc = [x + y for x, y in zip(desc1, desc2)]
    return_dict = {
        'KeyPoint' : dict1['KeyPoint'],
        'Descriptor' : sumDesc
    }
    return return_dict

def divide_dict(dict, number):
    divide_dict = [x / number for x in dict]
    return divide_dict

def crop_object(img):
    grayScale = to_gray(img)
    grayScale = cv2.inRange(grayScale, 0, 20)
    cnts = cv2.findContours(grayScale.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    returnImage = None
    # loop all contours to draw rectangle for moving object
    for c in cnts:
        (pX, pY, width, height) = cv2.boundingRect(c)
        # only proceed if the radius meets a minimum size
        if width * height > 5000:
            returnImage = img[pY:pY + height, pX:pX + width]
    return returnImage

def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    #s = ssim(imageA, imageB)

    # setup the figure
    fig = plt.figure(title)
    #plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    # show the images
    plt.show()