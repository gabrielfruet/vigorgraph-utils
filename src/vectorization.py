from functools import reduce
import logging
import atexit
import os
import cv2 as cv
import json
import glob
import argparse
from findunion import concatenate_lines
from utils import draw_line
from typing import Tuple, List
from algorithms import edge_linking, rdp
from preprocessing import find_background_color_range, find_background_mask
from skimage.util import img_as_ubyte
from skimage import morphology 
from seedlings import SeedlingSolver
import numpy as np

from yolo import YOLOProxy

atexit.register(cv.destroyAllWindows)

SHOW_IMAGE = False
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
    #raiz_prim_links_rdp = [link for link in raiz_prim_links_rdp if line_length(link) > 10]
    #hipocotilo_links_rdp = [link for link in hipocotilo_links_rdp if line_length(link) > 10]
    raiz_prim_links_rdp = concatenate_lines(raiz_prim_links_rdp, threshold=15)
    hipocotilo_links_rdp = concatenate_lines(hipocotilo_links_rdp, threshold=15)
    return raiz_prim_links_rdp, hipocotilo_links_rdp

def rm_bg(img: np.ndarray):
    color_range = find_background_color_range(img)
    color_range = find_background_color_range(img)
    input_img_wo_background = find_background_mask(img, color_range)
    return input_img_wo_background

def find_seed_blobs(input_img_wo_background: np.ndarray, iterations=5):
    input_img_wo_background = cv.erode(input_img_wo_background, np.ones((3,3)), iterations=iterations)
    return [cnt for cnt in cv.findContours(input_img_wo_background, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0] if cv.contourArea(cnt) < 1000]

def run(input_img_paths):
    model = YOLOProxy('./models/yolov8/best.pt')
    imgs = []
    for img_path in input_img_paths:
        imgs.append(cv.imread(img_path))

    logging.getLogger().setLevel(logging.WARNING)

    raiz_prim_masks, hipocotilo_masks = model.predict(imgs, imgsz=1280)
    output = dict()
    for hipocotilo_mask, raiz_prim_mask, img, im_path in zip(raiz_prim_masks, hipocotilo_masks, imgs, input_img_paths):
        raiz_prim_links, hipocotilo_links = find_lines(raiz_prim_mask, hipocotilo_mask, epsilon=1)
        input_img_wo_background = rm_bg(img) 
        contours  = find_seed_blobs(input_img_wo_background, iterations=8)
        cotyledone = []
        for ct in contours:
            M = cv.moments(ct)
            cX = int(M["m10"] / (M["m00"] + 0.001))
            cY = int(M["m01"] / (M["m00"] + 0.001))
            cotyledone.append((cX,cY))

        ss = SeedlingSolver(raiz_prim_links, hipocotilo_links, np.array(cotyledone), max_cost=200)
        seedlings = ss.match()
        
        info = {
            'links': {
                'hipocotilo': [sdl.hipocotilo.tolist() for sdl in seedlings if sdl.hipocotilo is not None],
                'raiz_prim': [sdl.raiz_prim.tolist() for sdl in seedlings if sdl.raiz_prim is not None],
            },
            'numero_plantulas': len(seedlings),
            'numero_plantuas_ngerm': reduce(lambda acc, sdl: int(sdl.is_dead()) + acc, seedlings, 0) 
        }
        output[im_path] = info

    """
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
    """
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('path', type=str, help='Path to the data file')
    parser.add_argument('--batch_size', type=int, default=5, help='Size of each batch')

    # Parse arguments
    args = parser.parse_args()

    files = [os.path.abspath(path) for path in glob.glob('./dataset/1/input/*')]
    n = len(files)
    batch_size = args.batch_size
    results = []
    for i in range(0, n, batch_size):
        result = run(files[i: min(i+batch_size, n)])
        results.append(result)
        
    merged_result = reduce(lambda x, y: x | y, results)
    print(json.dumps(merged_result))
