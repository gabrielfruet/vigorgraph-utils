import logging
import atexit
from pprint import pprint
import cv2 as cv
from algorithms import edge_linking, rdp
from preprocessing import split_image, apply_skeletonize, find_background_color_range, find_background_mask
import numpy as np

atexit.register(cv.destroyAllWindows)

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
        j,i = pts[i]
        img[i,j] = (255,255,255)

SHOW_IMAGE = True

def run():
    logging.getLogger().setLevel(logging.WARNING)
    gd_img = cv.imread(GD_IMG_PATH)
    input_img = cv.imread(INPUT_IMG_PATH)

    raiz_prim, hipocotilo = split_image(gd_img)
    raiz_prim = apply_skeletonize(raiz_prim)
    hipocotilo = apply_skeletonize(hipocotilo)
    raiz_prim_links = edge_linking(raiz_prim)
    hipocotilo_links = edge_linking(hipocotilo)

    linked_raiz_prim = cv.cvtColor(np.zeros_like(raiz_prim), cv.COLOR_GRAY2BGR)
    linked_hipocotilo = cv.cvtColor(np.zeros_like(hipocotilo), cv.COLOR_GRAY2BGR)

    raiz_prim_links_rdp = [rdp(link,epsilon=5) for link in raiz_prim_links]
    hipocotilo_links_rdp = [rdp(link,epsilon=5) for link in hipocotilo_links]
    
    for rdplink in raiz_prim_links_rdp:
        draw_lines(linked_raiz_prim, rdplink, color=(0,255,0))

    for rdplink in hipocotilo_links_rdp:
        draw_lines(linked_hipocotilo, rdplink, color=(0,0,255))

    color_range = find_background_color_range(input_img)

    input_img_wo_background = find_background_mask(input_img, color_range)
    input_img_wo_background = cv.erode(input_img_wo_background, np.ones((15,15)))
    contours, hierarchy = cv.findContours(input_img_wo_background, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    input_img_wo_background = cv.cvtColor(input_img_wo_background, cv.COLOR_GRAY2BGR)

    cv.drawContours(input_img_wo_background, contours, -1, (255,0,255), 0);

    output_json = {
        'Links': {
            'Hipocotilo': [link.tolist() for link in hipocotilo_links_rdp],
            'Raiz Primaria': [link.tolist() for link in raiz_prim_links_rdp]
        }
    }
    pprint(output_json)

    if SHOW_IMAGE:
        while True:
            cv.imshow('raiz_prim', raiz_prim)
            cv.imshow('hip', hipocotilo)
            cv.imshow('linked_raiz_prim', linked_raiz_prim)
            cv.imshow('linked_hipocotilo', linked_hipocotilo)
            cv.imshow('input', input_img)

            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                break

if __name__ == '__main__':
    run()
