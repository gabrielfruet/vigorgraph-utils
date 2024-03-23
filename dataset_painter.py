from cv2.typing import MatLike
from pprint import pprint
import numpy as np
import cv2
import glob
import os

print = pprint

mode = False

class SeedlingDrawer:
    drawing = False  # True if mouse is pressed
    color = (0, 0, 255)  # Default to red
    erase = False  # True if erasing

    @staticmethod
    def create_drawer_callback(img):
        # Mouse callback function
        def draw(event, x, y, flags, param):
            new_x = x
            new_y = y

            if event == cv2.EVENT_LBUTTONDOWN:
                SeedlingDrawer.drawing = True
            elif event == cv2.EVENT_MOUSEMOVE:
                if SeedlingDrawer.drawing:
                    if SeedlingDrawer.erase:
                        cv2.circle(img, (new_x, new_y), 5, (0, 0, 0), -1)
                    else:
                        cv2.circle(img, (new_x, new_y), 1, SeedlingDrawer.color, -1)
            elif event == cv2.EVENT_LBUTTONUP:
                SeedlingDrawer.drawing = False
                if SeedlingDrawer.erase:
                    cv2.circle(img, (new_x, new_y), 10, (0, 0, 0), -1)
                else:
                    cv2.circle(img, (new_x, new_y), 1, SeedlingDrawer.color, -1)

        return draw

    @staticmethod
    def image_drawer(pathname, proportion=0.3) -> tuple[MatLike, MatLike] | None:
        """
        Display an image and enable drawing on it with mouse interactions.

        This function opens a window showing the input image at a specified proportion of its original size.
        Users can interact with the image using the mouse to draw, and keyboard inputs to change colors and
        toggle between drawing and erasing modes. Drawing is done on a semi-transparent overlay atop the resized
        image. The function relies on the `SeedlingDrawer` class for some global settings like color and erasing
        mode.

        Parameters

        - pathname : str
            The file path to the input image.
        - proportion : float, optional
            The proportion to resize the image, with the default being 0.5 (50%).

        Key Bindings

        - 'r': Set the drawing color to RAIZ PRIMARIA.
        - 'g': Set the drawing color to HIPOCOTILO.
        - 'd': Activate drawing mode.
        - 'e': Activate erasing mode.
        - 'x': Terminate the draw and return None.
        - 's': Exit the drawing window.

        Returns

        - tuple of numpy.ndarray
            A tuple containing two elements; the first is the resized input image, and the second is the overlay image with drawings.

        Note:
        The function makes use of global state within the `SeedlingDrawer` class, which may lead to unexpected behavior if used concurrently in multiple instances.
        """
        input_image = cv2.imread(pathname)
        overlay = np.zeros_like(input_image)

        h, w, _ = input_image.shape

        nh = int(proportion * h)
        nw = int(proportion * w)

        resized_input_image = cv2.resize(input_image, (nw, nh), interpolation=cv2.INTER_AREA)
        resized_overlay = cv2.resize(overlay, (nw, nh), interpolation=cv2.INTER_AREA)

        draw = SeedlingDrawer.create_drawer_callback(resized_overlay)

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw)

        while True:
            blended = cv2.addWeighted(resized_input_image, 0.5, resized_overlay, 1 - 0.5, 0)
            cv2.imshow('image', blended)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('h'):
                print("Drawing HIPOCOTILO")
                SeedlingDrawer.color = (0, 0, 255)  # Red
                SeedlingDrawer.erase = False
            elif k == ord('r'):
                print("Drawing RAIZ PRIMARIA")
                SeedlingDrawer.color = (0, 255, 0)  # Green
                SeedlingDrawer.erase = False
            elif k == ord('d'):
                print("Drawing")
                SeedlingDrawer.erase = False  # Activate drawing
            elif k == ord('e'):
                print("Erasing")
                SeedlingDrawer.erase = True  # Activate erasing
            elif k == ord('x'):
                print("Exiting WITHOUT saving the image")
                cv2.destroyAllWindows()
                return None
            elif k == ord('s'):  # ESC key to exit
                print("Save and go to the next IMAGE")
                break

        cv2.destroyAllWindows()
        return resized_input_image, resized_overlay


class SeedlingDataset:

    def __init__(self, in_folder, out_folder, *, extension='jpg', in_suffix='input', out_suffix='ground_truth', redo=False):
        self.INPUT_FOLDER = in_folder
        self.OUTPUT_FOLDER = out_folder
        self.EXTENSION = extension
        self.IN_SUFFIX = in_suffix
        self.OUT_SUFFIX = out_suffix
        self.REDO = redo

    def make_directories(self):
        os.makedirs(os.path.join(self.OUTPUT_FOLDER), exist_ok=True)
        for folder in [self.IN_SUFFIX, self.OUT_SUFFIX]:
            os.makedirs(os.path.join(self.OUTPUT_FOLDER, folder), exist_ok=True)


    def already_done(self):
        return set(map(
            lambda x: self.INPUT_FOLDER + x.replace(f'_{self.IN_SUFFIX}.{self.EXTENSION}', '') + '.' + self.EXTENSION,
            glob.glob(os.path.join(self.OUTPUT_FOLDER, f'*_{self.IN_SUFFIX}.{self.EXTENSION}'))
        ))

    def run(self):
        expandable_path = os.path.join(self.INPUT_FOLDER, '*')
        images_path = set(glob.glob(expandable_path))
        self.make_directories()
        for img_path in images_path:
            img_name = os.path.split(img_path)[1]

            io = SeedlingDrawer.image_drawer(img_path)

            if io is None:
                return

            in_img, out_img = io

            in_img_path = os.path.join(self.OUTPUT_FOLDER, f'{img_name}_{self.IN_SUFFIX}.{self.EXTENSION}')
            out_img_path = os.path.join(self.OUTPUT_FOLDER, f'{img_name}_{self.OUT_SUFFIX}.{self.EXTENSION}')

            cv2.imwrite(in_img_path, in_img)
            cv2.imwrite(out_img_path, out_img)


input_folder = './plantulas_soja/1'
output_folder = './dataset/plantulas_soja/1'

sd = SeedlingDataset(input_folder, output_folder)
sd.run()
