# import the necessary packages
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

# image1 = cv2.imread('../sample_img/020298.png')
# image2 = cv2.imread('../sample_img/020299.png')
# image3 = cv2.imread('../sample_img/020300.png')
# image4 = cv2.imread('../sample_img/020301.png')
# image5 = cv2.imread('../sample_img/020302.png')
# image6 = cv2.imread('../sample_img/020303.png')
# image7 = cv2.imread('../sample_img/test1_crop.jpg')
# image8 = cv2.imread('../sample_img/test2_crop.jpg')
# image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# print(cv2.HuMoments(cv2.moments(image)).flatten())

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



# image7 = crop_object(image7)
# image8 = crop_object(image8)
# cv2.imshow('1', image1)
# cv2.imshow('2', image2)
# cv2.imshow('3', image3)
# cv2.imshow('4', image4)
# cv2.imshow('5', image5)
# cv2.imshow('6', image6)
# cv2.imshow('7', image7)
# cv2.imshow('8', image8)
# print('1-------------')
# print(momentColorExtraction(image1))
# print('2-------------')
# print(momentColorExtraction(image2))
# print('3-------------')
# print(momentColorExtraction(image3))
# print('4-------------')
# print(momentColorExtraction(image4))
# print('5-------------')
# print(momentColorExtraction(image5))
# print('6-------------')
# print(momentColorExtraction(image6))
# print('7-------------')
# print(momentColorExtraction(image7))
# print('8-------------')
# print(momentColorExtraction(image8))
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
    s = ssim(imageA, imageB)

    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

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

# # load the images -- the original, the original + contrast,
# # and the original + photoshop
# original = cv2.imread("../sample_img/test1_crop.jpg")
# contrast = cv2.imread("../sample_img/test1_crop.jpg")
# shopped = cv2.imread("../sample_img/test2_crop.jpg")
#
# # convert the images to grayscale
# original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
# contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
# shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)
# # initialize the figure
# fig = plt.figure("Images")
# images = ("Original", original), ("Contrast", contrast), ("Photoshopped", shopped)
#
# # loop over the images
# for (i, (name, image)) in enumerate(images):
#     # show the image
#     ax = fig.add_subplot(1, 3, i + 1)
#     ax.set_title(name)
#     plt.imshow(image, cmap=plt.cm.gray)
#     plt.axis("off")
#
# # show the figure
# plt.show()
#
# # compare the images
# compare_images(original, original, "Original vs. Original")
# compare_images(original, contrast, "Original vs. Contrast")
# compare_images(original, shopped, "Original vs. Photoshopped")
# plt.show()
#
# print(27**(1/3))