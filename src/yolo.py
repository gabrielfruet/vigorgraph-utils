from ultralytics import YOLO 
import cv2 as cv
import numpy as np

class YOLOProxy:
    def __init__(self, *args, **kwargs):
        self.yolo = YOLO(*args, **kwargs)

    def predict(self, imgs, **kwargs):
        results = self.yolo.predict(imgs, verbose=False, **kwargs)
        hipocotilos_masks = [np.zeros_like(img_i) for img_i in imgs]
        raizes_prim_masks = [np.zeros_like(img_i) for img_i in imgs]
        for i, result in enumerate(results):
            height,width,_ = imgs[i].shape
            for mask, class_id in zip(result.masks, result.boxes.cls):
                xyn = mask.xyn[0]
                xyn[:, 0] = xyn[:, 0]*width
                xyn[:, 1] = xyn[:, 1]*height
                xyn = np.array(xyn, dtype=np.int32)
                to_apply = hipocotilos_masks[i] if class_id == 0 else raizes_prim_masks[i]
                cv.drawContours(to_apply, xyn[None],0, color=(255,255,255), thickness=-1)

            hipocotilos_masks[i] = cv.cvtColor(hipocotilos_masks[i], cv.COLOR_BGR2GRAY) 
            raizes_prim_masks[i] = cv.cvtColor(raizes_prim_masks[i], cv.COLOR_BGR2GRAY)

            hipocotilos_masks[i] = cv.dilate(hipocotilos_masks[i], np.ones((3,3)), iterations=1)
            raizes_prim_masks[i] = cv.dilate(raizes_prim_masks[i], np.ones((3,3)), iterations=1)

        return raizes_prim_masks, hipocotilos_masks


