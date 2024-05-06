import cv2 as cv
import numpy as np
from preprocessing import split_image

GD_IMG_PATH = "/home/gabrielfruet/dev/python/vigorgraph/dataset/plantulas_soja/1/ground_truth/C2T1R1.jpg"

if __name__ == '__main__':
    gdimg = cv.imread(GD_IMG_PATH) 
    hipocotilo,_ = split_image(gdimg)
    noise = (np.random.randn(*hipocotilo.shape) > 1)
    hipocotilo = hipocotilo*noise
    
    k = np.zeros((31,31)).astype(np.uint8)
    k[:, 14:18] = 1

    hipocotilo1 = cv.dilate(hipocotilo, np.ones((31,31)))
    hipocotilo2 = cv.dilate(hipocotilo, k)

    while True:
        cv.imshow('hipocotilo noise', hipocotilo)
        cv.imshow('hipocotilo k normal', hipocotilo1)
        cv.imshow('hipocotilo k line', hipocotilo2)
        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            break
