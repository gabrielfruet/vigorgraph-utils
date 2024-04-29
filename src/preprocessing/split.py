import numpy as np
import cv2 as cv

GREEN_HSV_MIN = np.array((50, 120, 120))
GREEN_HSV_MAX = np.array((70, 255, 255))

def find_background_mask(img, color_range):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, color_range[0], color_range[1])
    #foreground = cv.bitwise_and(img, img, mask=mask)
    background = cv.bitwise_not(mask)
    #white_background = np.full(img.shape, 255, dtype=np.uint8)
    #bk_combined = cv.bitwise_and(white_background, white_background, mask=background)
    #final_image = cv.add(foreground, bk_combined)
    return background

def split_image(img):
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    green_mask = cv.inRange(hsv_img, GREEN_HSV_MIN, GREEN_HSV_MAX)
    gray_img = hsv_img[:,:,2]
    _, red_mask = cv.threshold(cv.bitwise_and(gray_img, cv.bitwise_not(green_mask)), 200, 255, cv.THRESH_BINARY)
    return green_mask, red_mask

def find_background_color_range(img, border_size=10):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    top_border = hsv_image[:border_size, :].reshape((-1,3))
    bottom_border = hsv_image[-border_size:, :].reshape((-1,3))
    left_border = hsv_image[:, :border_size].reshape((-1,3))
    right_border = hsv_image[:, -border_size:].reshape((-1,3))

    borders = np.r_[top_border, bottom_border, left_border, right_border]

    average_hsv = np.mean(borders, axis=0)
    avg_hue = np.mean(average_hsv[0])
    avg_sat = np.mean(average_hsv[1])
    avg_val = np.mean(average_hsv[2])

    hue_range = 40
    sat_range = 80
    val_range = 80

    lower_bound = np.array([max(avg_hue - hue_range, 0), max(avg_sat - sat_range, 0), max(avg_val - val_range, 0)], dtype=np.uint8)
    upper_bound = np.array([min(avg_hue + hue_range, 179), min(avg_sat + sat_range, 255), min(avg_val + val_range, 255)], dtype=np.uint8)

    return (lower_bound, upper_bound)
