import logging
import atexit
import cv2 as cv
import json
from utils import line_length
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
GD_IMG_PATH = "/home/gabrielfruet/dev/python/vigorgraph/dataset/plantulas_soja/cultivar_2_azul/ground_truth/1712091204433.jpg"
INPUT_IMG_PATH = "/home/gabrielfruet/dev/python/vigorgraph/dataset/plantulas_soja/cultivar_2_azul/input/1712091204433.jpg"
WEIGHTS_PATH = "/home/gabrielfruet/dev/python/vigorgraph/models/model(1).keras"

def preprocessing(img):
    transformed = img.copy()
    first_erode_kernel = np.ones((3, 3), np.uint8)

    second_erode_kernel = np.zeros((5, 5), np.uint8)
    second_erode_kernel[2, :] = 1

    dilate_kernel = np.zeros((5,5), np.uint8)
    dilate_kernel[:, 2] = 1

    #transformed = cv.erode(transformed, second_erode_kernel, iterations=1)
    transformed = pcv.fill(transformed, size=50)
    transformed = cv.dilate(transformed, dilate_kernel, iterations=2)
    #transformed = cv.morphologyEx(transformed, cv.MORPH_OPEN, np.ones((3,3)), iterations=3)

    #for _ in range(5):
    #transformed = cv.morphologyEx(transformed, cv.MORPH_CLOSE, np.ones((3,3)), iterations=3)


    #transformed = cv.morphologyEx(transformed, cv.MORPH_OPEN, np.ones((3,3)), iterations=5)
    #transformed = cv.morphologyEx(transformed, cv.MORPH_CLOSE, np.ones((3,3)), iterations=5)

    #transformed = cv.morphologyEx(transformed, cv.MORPH_OPEN, np.ones((3,3)), iterations=2)
    #transformed = cv.dilate(transformed, dilate_kernel, iterations=1)

    #transformed = cv.dilate(transformed, np.ones((3,3)), iterations=3)

    return transformed

def paint_links(img, links, color):
    for link in links:
        for pos in link:
            img[*pos] = color
    return img

def draw_lines(img, pts, color, pt_color):
    n = len(pts)
    for i in range(n-1):
        cv.line(img, pts[i], pts[i+1], color=color)

    for i in range(n):
        cv.circle(img, pts[i], 1, pt_color, 1)

def find_lines(raiz_prim: np.ndarray, hipocotilo: np.ndarray, epsilon=5) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """

    Returns

    - tuple of numpy.ndarray
        1st is the Raiz Primaria links
        2nd is the Hipocotilo links
    """
    raiz_prim = img_as_ubyte(pcv.morphology.prune(pcv.morphology.skeletonize(raiz_prim), size=10)[0])
    hipocotilo = img_as_ubyte(pcv.morphology.prune(pcv.morphology.skeletonize(hipocotilo), size=10)[0])
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
def find_seed_blobs(input_img_wo_background: np.ndarray, iterations=5):
    input_img_wo_background = cv.erode(input_img_wo_background, np.ones((3,3)), iterations=iterations)
    return cv.findContours(input_img_wo_background, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

def run():
    logging.getLogger().setLevel(logging.WARNING)
    gd_img = cv.imread(GD_IMG_PATH)
    input_img = cv.imread(INPUT_IMG_PATH)

    raiz_prim, hipocotilo = split_image(gd_img)
    raiz_prim_links_rdp, hipocotilo_links_rdp = find_lines(raiz_prim, hipocotilo, epsilon=5)
    print(len(raiz_prim_links_rdp))
    print(len(hipocotilo_links_rdp))

    linked_raiz_prim = np.zeros_like(gd_img)
    linked_hipocotilo = np.zeros_like(gd_img)

    input_img_wo_background = rm_bg(input_img)
    contours, _ = find_seed_blobs(input_img_wo_background, iterations=7)
    input_img_wo_background = cv.cvtColor(input_img_wo_background, cv.COLOR_GRAY2BGR)
    input_img_wo_background[:,:,:] = (0,0,0)

    cv.drawContours(input_img_wo_background, contours, -1, (255,0,255), 0);
    cotyledone = []
    for ct in contours:
        M = cv.moments(ct)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cotyledone.append((cX,cY))

        input_img_wo_background = cv.circle(input_img_wo_background, (cX, cY), radius=5, color=(0,255,255), thickness=-1)

    ss = SeedlingSolver(raiz_prim_links_rdp, hipocotilo_links_rdp, np.array(cotyledone))
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
    raiz_prim_skeleton = pcv.morphology.prune(pcv.morphology.skeletonize(raiz_prim))[0]
    hip_skeleton = pcv.morphology.prune(pcv.morphology.skeletonize(hipocotilo))[0]

    if SHOW_IMAGE:
        while True:
            cv.imshow('seedlings_drawed', seedlings_drawed)
            cv.imshow('raiz_prim_ske', raiz_prim_skeleton)
            cv.imshow('hip_ske', hip_skeleton)
            cv.imshow('linked_raiz_prim', linked_raiz_prim)
            cv.imshow('linked_hipocotilo', linked_hipocotilo)
            cv.imshow('seed blobs', input_img_wo_background)

            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                break

"""

def with_model():
    model = create_model()
    model.load_weights(WEIGHTS_PATH)
    input_img = cv.imread(INPUT_IMG_PATH)
    gd_img = cv.imread(GD_IMG_PATH)
    h,w,_ = input_img.shape

    model_input = np.asarray(img_as_float32(cv.resize(input_img, (224,224)))*2 - 1)
    model_input = np.expand_dims(model_input, axis=0)

    batched_prediction = model(model_input)

    predicted = np.argmax(batched_prediction[0], axis=-1)

    #plt.imshow(predicted)
    #plt.show()

    hipocotilo = cv.resize(img_as_ubyte(predicted == 1), (w,h), interpolation=cv.INTER_NEAREST)
    raiz_prim = cv.resize(img_as_ubyte(predicted == 2), (w,h), interpolation=cv.INTER_NEAREST)

    hipP = preprocessing(hipocotilo)
    raizP = preprocessing(raiz_prim)

    raiz_overlay = np.zeros_like(input_img)
    raiz_overlay[raiz_prim == 255] = [255,0,0]
    raiz_overlay[raizP == 255] = [0,0,255]

    raiz_ske = img_as_ubyte(skeletonize(raizP))
    raiz_ske_pru = pcv.morphology.prune(raiz_ske, size=50)[0]

    if SHOW_IMAGE:
        while True:
            cv.imshow('input_img', input_img)
            cv.imshow('gdimg', gd_img)
            #cv.imshow('hip_pred', hipocotilo)
            #cv.imshow('raiz_pred', raiz_prim)
            #cv.imshow('hipP', hipP)
            #cv.imshow('raizP', raizP)
            cv.imshow('hip_overlay', raiz_overlay)
            cv.imshow('hip_ske', raiz_ske)
            cv.imshow('hip_ske_prun', raiz_ske_pru)

            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                break
"""

if __name__ == '__main__':
    run()
