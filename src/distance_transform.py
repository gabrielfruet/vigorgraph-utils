from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte, img_as_float, img_as_bool
import cv2 as cv
import numpy as np
import atexit
from plantcv import plantcv as pcv
from skimage import measure
import skimage
from skimage.morphology import medial_axis

atexit.register(cv.destroyAllWindows)
GD_IMG_PATH = "/home/gabrielfruet/dev/python/vigorgraph/dataset/plantulas_soja/1/ground_truth/C2T1R4.jpg"
INPUT_IMG_PATH = "/home/gabrielfruet/dev/python/vigorgraph/dataset/plantulas_soja/1/input/C2T1R4.jpg"

def click_and_zoom_factory(src):
    def click_and_zoom(event, x, y, flags, param, src=src):
        if event == cv.EVENT_LBUTTONDOWN:  # Check for left mouse click
            # Define the size of the window to zoom into. This will be a square region.
            window_size = 100  # Size of the square region to zoom in on

            # Calculate the coordinates of the region around the click point
            # Ensure that the region stays within the bounds of the image
            start_x = max(x - window_size // 2, 0)
            start_y = max(y - window_size // 2, 0)
            end_x = min(x + window_size // 2, src.shape[1] - 1)
            end_y = min(y + window_size // 2, src.shape[0] - 1)

            # Extract the region of interest (ROI) from the image
            roi = src[start_y:end_y, start_x:end_x]

            # Resize the ROI back to the size of the original image using an interpolation method
            zoomed = cv.resize(roi, (src.shape[1], src.shape[0]), interpolation=cv.INTER_AREA)

            # Display the zoomed image in a new window
            while True:
                cv.imshow("Zoomed", zoomed)
                if cv.waitKey(50) == 27:
                    break

            cv.destroyWindow("Zoomed")

    return click_and_zoom

def my_bg_remover(img):
    lr,hr = np.percentile(img, [4, 96],axis=[0,1])
    bg_mask = cv.bitwise_not(cv.inRange(img,np.array(lr), np.array(hr)))
    result = pcv.fill(bg_mask, size=200)
    return result

def my_bg_remover2(img):
    r = cv.Canny(img, 20,250)
    r = cv.morphologyEx(r, cv.MORPH_CLOSE, np.ones((3,3)), iterations=10)
    return r

def show_image(window_name, image):
    if not isinstance(window_name, list):
        window_name = [window_name]
        image = [image]

    for i_window_name, i_img in zip(window_name, image):
        cv.namedWindow(i_window_name)
        cv.setMouseCallback(i_window_name, click_and_zoom_factory(i_img))
    while True:
        for i_window_name, i_img in zip(window_name, image):
            cv.imshow(i_window_name, i_img)
        if cv.waitKey(50) == 27:
            break

if __name__ == '__main__':
    img = cv.imread(INPUT_IMG_PATH)
    r = my_bg_remover2(img)

    show_image(
        ['img', 'r'],
        [img, r]
    )

    cv.destroyAllWindows()

