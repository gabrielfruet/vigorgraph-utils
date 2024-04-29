import logging
import atexit
import cv2 as cv
import json
from pprint import pprint
from typing import Tuple, List
from algorithms import edge_linking, rdp
from preprocessing import split_image, apply_skeletonize, find_background_color_range, find_background_mask
import numpy as np

atexit.register(cv.destroyAllWindows)

SHOW_IMAGE = True
GD_IMG_PATH = "/home/gabrielfruet/dev/python/vigorgraph/dataset/plantulas_soja/1/ground_truth/C2T1R1.jpg"
INPUT_IMG_PATH = "/home/gabrielfruet/dev/python/vigorgraph/dataset/plantulas_soja/1/input/C2T1R1.jpg"

def paint_links(img, links, color):
    for link in links:
        for pos in link:
            img[*pos] = color
    return img

def draw_lines(img, pts, color):
    n = len(pts)
    for i in range(n-1):
        cv.line(img, pts[i], pts[i+1], color=color)

    for i in range(n):
        cv.circle(img, pts[i], 2, (255,255,255), 2)
        j,i = pts[i]
        img[i,j] = (255,255,255)

def find_lines(gd_img: np.ndarray, epsilon=5) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """

    Returns

    - tuple of numpy.ndarray
        1st is the Raiz Primaria links
        2nd is the Hipocotilo links
    """
    raiz_prim, hipocotilo = split_image(gd_img)
    raiz_prim = apply_skeletonize(raiz_prim)
    hipocotilo = apply_skeletonize(hipocotilo)
    raiz_prim_links = edge_linking(raiz_prim)
    hipocotilo_links = edge_linking(hipocotilo)
    raiz_prim_links_rdp = [rdp(link,epsilon=epsilon) for link in raiz_prim_links]
    hipocotilo_links_rdp = [rdp(link,epsilon=epsilon) for link in hipocotilo_links]
    return raiz_prim_links_rdp, hipocotilo_links_rdp

def rm_bg(img: np.ndarray):
    color_range = find_background_color_range(img)
    color_range = find_background_color_range(img)
    input_img_wo_background = find_background_mask(img, color_range)
    return input_img_wo_background

def find_seed_blobs(input_img_wo_background: np.ndarray):
    input_img_wo_background = cv.erode(input_img_wo_background, np.ones((15,15)))
    contours, _ = cv.findContours(input_img_wo_background, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    input_img_wo_background = cv.cvtColor(input_img_wo_background, cv.COLOR_GRAY2BGR)
    return contours

def run():
    logging.getLogger().setLevel(logging.WARNING)
    gd_img = cv.imread(GD_IMG_PATH)
    input_img = cv.imread(INPUT_IMG_PATH)

    raiz_prim_links_rdp, hipocotilo_links_rdp = find_lines(gd_img, epsilon=40)

    linked_raiz_prim = np.zeros_like(gd_img)
    linked_hipocotilo = np.zeros_like(gd_img)
    
    for rdplink in raiz_prim_links_rdp:
        draw_lines(linked_raiz_prim, rdplink, color=(0,255,0))

    for rdplink in hipocotilo_links_rdp:
        draw_lines(linked_hipocotilo, rdplink, color=(0,0,255))

    input_img_wo_background = rm_bg(input_img)
    contours = find_seed_blobs(input_img_wo_background)

    cv.drawContours(input_img_wo_background, contours, -1, (255,0,255), 0);

    output = {
        'Links': {
            'Hipocotilo': [link.tolist() for link in hipocotilo_links_rdp],
            'Raiz Primaria': [link.tolist() for link in raiz_prim_links_rdp]
        }
    }
    json_output = json.dumps(output)
    pprint(json_output)

    if SHOW_IMAGE:
        while True:
            cv.imshow('linked_raiz_prim', linked_raiz_prim)
            cv.imshow('linked_hipocotilo', linked_hipocotilo)
            cv.imshow('seed blobs', input_img_wo_background)

            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                break

if __name__ == '__main__':
    run()
