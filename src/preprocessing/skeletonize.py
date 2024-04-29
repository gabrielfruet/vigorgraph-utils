import cv2 as cv
import numpy as np
from skimage.util import img_as_ubyte, img_as_bool
from skimage.morphology import skeletonize

def apply_skeletonize(gray_img):
    dilated_gray_img = cv.dilate(gray_img, np.ones((3, 3)))
    float_img = img_as_bool(dilated_gray_img)
    skeleton = skeletonize(float_img)
    skeleton_ubyte = img_as_ubyte(skeleton)
    return skeleton_ubyte
