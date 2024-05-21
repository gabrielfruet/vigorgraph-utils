from ultralytics import YOLO 
import cv2 as cv
import numpy as np

class YOLOProxy:
    def __init__(self, *args, **kwargs):
        self.yolo = YOLO(*args, **kwargs)

    def predict(self, image_path, **kwargs):
        img = cv.imread(image_path)
        results = self.yolo.predict(image_path, **kwargs)
        hipocotilos_mask = np.zeros_like(img)
        raizes_prim_mask = np.zeros_like(img)
        height,width,_ = img.shape
        for result in results:
            for mask, class_id in zip(result.masks, result.boxes.cls):
                xyn = mask.xyn[0]
                xyn[:, 0] = xyn[:, 0]*width
                xyn[:, 1] = xyn[:, 1]*height
                xyn = np.array(xyn, dtype=np.int32)
                to_apply = hipocotilos_mask if class_id == 0 else raizes_prim_mask
                cv.drawContours(to_apply, xyn[None],0, color=(255,255,255), thickness=-1)

        while True:
            cv.imshow('hipocotilo', hipocotilos_mask)
            cv.imshow('raiz', raizes_prim_mask)
            k = cv.waitKey(1000)
            if ord('q') == k: break

        hipocotilos_mask = cv.cvtColor(hipocotilos_mask, cv.COLOR_BGR2GRAY) 
        raizes_prim_mask = cv.cvtColor(raizes_prim_mask, cv.COLOR_BGR2GRAY)

        return raizes_prim_mask, hipocotilos_mask


