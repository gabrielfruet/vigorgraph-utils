import cv2 as cv
import skimage as ski
from matplotlib import pyplot as plt
from skimage.util import img_as_ubyte, img_as_float, img_as_bool
from skimage.morphology import skeletonize
import numpy as np
import atexit

atexit.register(cv.destroyAllWindows)

GD_IMG_PATH = "/home/gabrielfruet/dev/python/vigorgraph/dataset/plantulas_soja/1/ground_truth/C2T1R1.jpg"
INPUT_IMG_PATH = "/home/gabrielfruet/dev/python/vigorgraph/dataset/plantulas_soja/1/input/C2T1R1.jpg"
GREEN_HSV_MIN = np.array((50, 120, 120))
GREEN_HSV_MAX = np.array((70, 255, 255))

def split_image(img):
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    green_mask = cv.inRange(hsv_img, GREEN_HSV_MIN, GREEN_HSV_MAX)
    gray_img = hsv_img[:,:,2]
    _, red_mask = cv.threshold(cv.bitwise_and(gray_img, cv.bitwise_not(green_mask)), 200, 255, cv.THRESH_BINARY)
    return green_mask, red_mask

def apply_skeletonize(gray_img):
    dilated_gray_img = cv.dilate(gray_img, np.ones((3, 3)))
    float_img = img_as_bool(dilated_gray_img)
    skeleton = skeletonize(float_img)
    skeleton_ubyte = img_as_ubyte(skeleton)
    return skeleton_ubyte

def edge_linking(A: np.ndarray):
    """

    A is a np.ndarray with 0s, and 1s

    """
    A = A.copy()
    links = []
    n,m = A.shape
    def neighbours(i,j):
        pos = np.array([i,j])
        directions = np.array([[1,-1], [1,0], [1,1],[0,-1],[0,1],[-1,-1],[-1,0],[-1,1]])
        valid_neighbours = []
        for dir in directions:
            new_pos = dir + pos
            if np.all(new_pos < [n,m]) and np.all(new_pos >= [0,0]):
                valid_neighbours.append(new_pos)
        
        return valid_neighbours

    for i in range(n):
        for j in range(m):
            if A[i,j] == 255:
                A[i,j] = 0
                k,l = i,j
                new_link = [(i,j)]
                available_paths=True
                while available_paths:
                    for p,q in neighbours(k,l):
                        if A[p,q] == 255:
                            A[p,q] = 0
                            new_link.append((p,q))
                            k,l = p,q
                            break
                    else:
                        available_paths = False

                links.append(np.array(new_link))

    return links

def paint_links(img, links, color):
    for link in links:
        for pos in link:
            img[*pos] = color
    return img

def ramer_douglas_peucker(points, epsilon) -> np.ndarray:
    if len(points) < 3: return points
    first, last = points[[0,-1]]
    x0,y0 = first
    x1,y1 = last

    delta_x, delta_y = last - first
    m = delta_y/delta_x
    a = m
    b = -1
    c = y1 - m*x1

    distances = np.abs(np.dot(points, [a,b]) + c) /  np.sqrt(a**2 + b**2)
    max_i = np.argmax(distances)
    max_dist = distances[max_i]

    if max_dist > epsilon:
        result1 = ramer_douglas_peucker(points[0:max_i], epsilon)
        result2 = ramer_douglas_peucker(points[max_i:-1], epsilon)
        return np.r_[result1, result2]
    else:
        return points[[0,-1]]

def simple_link(link, ax=plt):
    linkx = link[:, 0]
    linky = link[:, 1]

    freq = np.fft.fftfreq(len(linky), 1)
    #ax.plot(freq, np.fft.fft(linky))

    ax.plot(linkx)

def draw_lines(img, pts, color):
    n = len(pts)
    for i in range(n-1):
        cv.line(img, pts[i, ::-1], pts[i+1, ::-1], color=color)

    for i in range(n):
        img[*pts[i]] = (203, 192, 255)

def find_background_mask(img, color_range):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, color_range[0], color_range[1])
    foreground = cv.bitwise_and(img, img, mask=mask)
    background = cv.bitwise_not(mask)
    white_background = np.full(img.shape, 255, dtype=np.uint8)
    bk_combined = cv.bitwise_and(white_background, white_background, mask=background)
    final_image = cv.add(foreground, bk_combined)
    return background

def find_background_color_range(img, border_size=10):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    top_border = hsv_image[:border_size, :].reshape((-1,3))
    bottom_border = hsv_image[-border_size:, :].reshape((-1,3))
    left_border = hsv_image[:, :border_size].reshape((-1,3))
    right_border = hsv_image[:, -border_size:].reshape((-1,3))

    # Combine border pixels into one array
    borders = np.r_[top_border, bottom_border, left_border, right_border]

    # Calculate the average HSV value of the borders
    average_hsv = np.mean(borders, axis=0)
    avg_hue = np.mean(average_hsv[0])
    avg_sat = np.mean(average_hsv[1])
    avg_val = np.mean(average_hsv[2])

    # Define a reasonable range around the average
    hue_range = 40
    sat_range = 80
    val_range = 80

    lower_bound = np.array([max(avg_hue - hue_range, 0), max(avg_sat - sat_range, 0), max(avg_val - val_range, 0)], dtype=np.uint8)
    upper_bound = np.array([min(avg_hue + hue_range, 179), min(avg_sat + sat_range, 255), min(avg_val + val_range, 255)], dtype=np.uint8)

    return (lower_bound, upper_bound)

SHOW_IMAGE = True

def run():
    gd_img = cv.imread(GD_IMG_PATH)
    input_img = cv.imread(INPUT_IMG_PATH)

    raiz_prim, hipocotilo = split_image(gd_img)
    raiz_prim = apply_skeletonize(raiz_prim)
    hipocotilo = apply_skeletonize(hipocotilo)
    raiz_prim_links = edge_linking(raiz_prim)
    hipocotilo_links = edge_linking(hipocotilo)

    linked_raiz_prim = cv.cvtColor(np.zeros_like(raiz_prim), cv.COLOR_GRAY2BGR)
    linked_hipocotilo = cv.cvtColor(np.zeros_like(hipocotilo), cv.COLOR_GRAY2BGR)

    """
    fig, ax = plt.subplots(1)
    linkx, linky = links[0][:,0], links[0][:,1]
    resultx, resulty = result[:,0], result[:,1]

    ax.plot(linkx,linky,c='b')
    ax.plot(resultx,resulty,c='r')
    plt.show()
    """
    for rdplink in map(lambda link: ramer_douglas_peucker(link,epsilon=5), raiz_prim_links):
        draw_lines(linked_raiz_prim, rdplink, color=(0,255,0))

    for rdplink in map(lambda link: ramer_douglas_peucker(link,epsilon=5), hipocotilo_links):
        draw_lines(linked_hipocotilo, rdplink, color=(0,0,255))

    color_range = find_background_color_range(input_img)

    input_img_wo_background = find_background_mask(input_img, color_range)
    input_img_wo_background = cv.erode(input_img_wo_background, np.ones((15,15)))
    contours, hierarchy = cv.findContours(input_img_wo_background, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    input_img_wo_background = cv.cvtColor(input_img_wo_background, cv.COLOR_GRAY2BGR)

    cv.drawContours(input_img_wo_background, contours, -1, (255,0,255), 0)

    summed = cv.bitwise_or(linked_hipocotilo, linked_raiz_prim)

    if SHOW_IMAGE:
        while True:
            cv.imshow('linked_raiz_prim', linked_raiz_prim)
            cv.imshow('linked_hipocotilo', linked_hipocotilo)
            cv.imshow('summed', summed)
            cv.imshow('no background input', input_img_wo_background)

            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                break

if __name__ == '__main__':
    run()