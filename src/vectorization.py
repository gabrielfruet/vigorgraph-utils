import logging
import atexit
import os
import cv2 as cv
import json
from findunion import concatenate_lines
from utils import line_length, draw_line
from skimage.morphology import skeletonize
from pprint import pprint
from typing import Tuple, List
from matplotlib import pyplot as plt
from algorithms import edge_linking, rdp
from preprocessing import split_image, apply_skeletonize, find_background_color_range, find_background_mask
from skimage.util import img_as_ubyte, img_as_float32
from skimage import morphology 
from plantcv import plantcv as pcv
from seedlings import SeedlingSolver
import numpy as np

atexit.register(cv.destroyAllWindows)

SHOW_IMAGE = True
DATASET_PATH = "/home/fruet/dev/python/vigorgraph-utils/dataset"
GD_IMG_PATH = "/home/fruet/dev/python/vigorgraph-utils/dataset/cultivar_5_azul/ground_truth/1712091560511.jpg"
INPUT_IMG_PATH = "/home/fruet/dev/python/vigorgraph-utils/dataset/cultivar_5_azul/input/1712091560511.jpg"
WEIGHTS_PATH = "/home/gabrielfruet/dev/python/vigorgraph/models/model(1).keras"

def draw_lines(img, pts, color, pt_color):
    draw_line(img,pts,color)
    for i in range(len(pts)):
        cv.circle(img, pts[i], 1, pt_color, 1)

def find_lines(raiz_prim: np.ndarray, hipocotilo: np.ndarray, epsilon=20) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """

    Returns

    - tuple of numpy.ndarray
        1st is the Raiz Primaria links
        2nd is the Hipocotilo links
    """
    raiz_prim = img_as_ubyte(morphology.skeletonize(raiz_prim))
    hipocotilo = img_as_ubyte(morphology.skeletonize(hipocotilo))
    raiz_prim_links = edge_linking(raiz_prim)
    hipocotilo_links = edge_linking(hipocotilo)
    raiz_prim_links_rdp = [rdp(link,epsilon=epsilon) for link in raiz_prim_links]
    hipocotilo_links_rdp = [rdp(link,epsilon=epsilon) for link in hipocotilo_links]
    raiz_prim_links_rdp = [link for link in raiz_prim_links_rdp if line_length(link) > 10]
    hipocotilo_links_rdp = [link for link in hipocotilo_links_rdp if line_length(link) > 10]
    raiz_prim_links_rdp = concatenate_lines(raiz_prim_links_rdp, threshold=10)
    hipocotilo_links_rdp = concatenate_lines(hipocotilo_links_rdp, threshold=10)
    return raiz_prim_links_rdp, hipocotilo_links_rdp

def rm_bg(img: np.ndarray):
    color_range = find_background_color_range(img)
    color_range = find_background_color_range(img)
    input_img_wo_background = find_background_mask(img, color_range)
    return input_img_wo_background

def find_seed_blobs(input_img_wo_background: np.ndarray, iterations=5):
    input_img_wo_background = cv.erode(input_img_wo_background, np.ones((3,3)), iterations=iterations)
    return [cnt for cnt in cv.findContours(input_img_wo_background, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0] if cv.contourArea(cnt) < 1000], input_img_wo_background

def run():
    logging.getLogger().setLevel(logging.WARNING)
    gd_img = cv.imread(GD_IMG_PATH)
    input_img = cv.imread(INPUT_IMG_PATH)

    raiz_prim, hipocotilo = split_image(gd_img)
    raiz_prim_links_rdp, hipocotilo_links_rdp = find_lines(raiz_prim, hipocotilo, epsilon=5)

    linked_raiz_prim = np.zeros_like(gd_img)
    linked_hipocotilo = np.zeros_like(gd_img)

    input_img_wo_background = rm_bg(input_img)
    contours, blobs_image = find_seed_blobs(input_img_wo_background, iterations=7)
    input_img_wo_background = cv.cvtColor(input_img_wo_background, cv.COLOR_GRAY2BGR)
    input_img_wo_background[:,:,:] = (0,0,0)

    cv.drawContours(input_img_wo_background, contours, -1, (255,0,255), 0);
    cotyledone = []
    print(len(contours))
    for ct in contours:
        M = cv.moments(ct)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cotyledone.append((cX,cY))

        input_img_wo_background = cv.circle(input_img_wo_background, (cX, cY), radius=5, color=(0,255,255), thickness=-1)

    ss = SeedlingSolver(raiz_prim_links_rdp, hipocotilo_links_rdp, np.array(cotyledone), max_cost=200)
    seedlings = ss.match()

    seedlings_drawed = np.zeros_like(input_img_wo_background)

    for sdl in seedlings:
        seedlings_drawed = sdl.draw(seedlings_drawed)
    
    for rdplink in raiz_prim_links_rdp:
        draw_lines(linked_raiz_prim, rdplink, color=(0,255,0), pt_color=(0,255,0))
        draw_lines(input_img_wo_background, rdplink, color=(0,255,0), pt_color=(0,255,0))

    for rdplink in hipocotilo_links_rdp:
        draw_lines(linked_hipocotilo, rdplink, color=(0,0,255), pt_color=(0,0,255))
        draw_lines(input_img_wo_background, rdplink, color=(0,0,255), pt_color=(0,0,255))

    output = {
        'links': {
            'hipocotilo': [link.tolist() for link in hipocotilo_links_rdp],
            'raiz_prim': [link.tolist() for link in raiz_prim_links_rdp]
        },
        'numero_plantulas': 20,
        'numero_plantuas_ngerm': 2
    }
    #json_output = json.dumps(output)
    #pprint(json_output)
    #raiz_prim_skeleton = pcv.morphology.prune(pcv.morphology.skeletonize(raiz_prim))[0]
    #hip_skeleton = pcv.morphology.prune(pcv.morphology.skeletonize(hipocotilo))[0]
    overlayed_img = cv.addWeighted(input_img, 0.5,seedlings_drawed, 0.5, 0)

    if SHOW_IMAGE:
        while True:
            cv.imshow('seedlings_drawed', seedlings_drawed)
            cv.imshow('overlay', overlayed_img)
            cv.imshow('blob', blobs_image)
            cv.imshow('blobs', blobs_image)
            cv.imshow('gd_img', gd_img)
            #cv.imshow('raiz_prim_ske', raiz_prim_skeleton)
            #cv.imshow('hip_ske', hip_skeleton)
            cv.imshow('linked_raiz_prim', linked_raiz_prim)
            cv.imshow('linked_hipocotilo', linked_hipocotilo)
            #cv.imshow('seed blobs', input_img_wo_background)

            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                break

if __name__ == '__main__':
    run()
