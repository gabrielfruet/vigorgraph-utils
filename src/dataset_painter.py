from cv2.typing import MatLike
from typing import Iterable, Set
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
    def create_drawer_callback(overlay_img, input_img, zoom_window_name, scale=3, img_size=384):
        # Mouse callback function
        def draw(event, x, y, flags, param):
            new_x = x
            new_y = y

            if event == cv2.EVENT_LBUTTONDOWN:
                SeedlingDrawer.drawing = True
            elif event == cv2.EVENT_MOUSEMOVE:
                
                if SeedlingDrawer.drawing:
                    if SeedlingDrawer.erase:
                        cv2.circle(overlay_img, (new_x, new_y), 5, (0, 0, 0), -1)
                    else:
                        cv2.circle(overlay_img, (new_x, new_y), 1, SeedlingDrawer.color, -1)

                if event == cv2.EVENT_MOUSEMOVE:
                    # Define the size of the zoomed window
                    zoom_size = img_size  # Zoom window will be 100x100 pixels
                    zoom_scale = scale   # Zoom scale

                    to_zoom_img = cv2.addWeighted(input_img, 0.5, overlay_img, 1 - 0.5, 0)

                    # Extract the sub-region of the image
                    top_left_x = max(0, x - zoom_size // (2 * zoom_scale))
                    top_left_y = max(0, y - zoom_size // (2 * zoom_scale))
                    bottom_right_x = min(to_zoom_img.shape[1], x + zoom_size // (2 * zoom_scale))
                    bottom_right_y = min(to_zoom_img.shape[0], y + zoom_size // (2 * zoom_scale))


                    sub_img = to_zoom_img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

                    # Resize sub-region to zoom in
                    sub_img_zoomed = cv2.resize(sub_img, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)

                    # Show zoomed image in new window
                    cv2.imshow(zoom_window_name, sub_img_zoomed)

            elif event == cv2.EVENT_LBUTTONUP:
                SeedlingDrawer.drawing = False
                if SeedlingDrawer.erase:
                    cv2.circle(overlay_img, (new_x, new_y), 10, (0, 0, 0), -1)
                else:
                    cv2.circle(overlay_img, (new_x, new_y), 1, SeedlingDrawer.color, -1)

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
        - 'h': Set the drawing color to HIPOCOTILO.
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
        print(f'new height: {nh}, new width: {nw}')

        resized_input_image = cv2.resize(input_image, (nw, nh), interpolation=cv2.INTER_AREA)
        resized_overlay = cv2.resize(overlay, (nw, nh), interpolation=cv2.INTER_AREA)

        draw = SeedlingDrawer.create_drawer_callback(resized_overlay, resized_input_image, 'zoomed')

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw)

        want_to_exit = False
        want_to_save = False

        while True:
            blended = cv2.addWeighted(resized_input_image, 0.5, resized_overlay, 1 - 0.5, 0)
            cv2.imshow('image', blended)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('h'):
                print("Drawing HIPOCOTILO")
                SeedlingDrawer.color = (0, 0, 255)  # Red
                SeedlingDrawer.erase = False
                want_to_exit = False
                want_to_save = False
            elif k == ord('r'):
                print("Drawing RAIZ PRIMARIA")
                SeedlingDrawer.color = (0, 255, 0)  # Green
                SeedlingDrawer.erase = False
                want_to_exit = False
                want_to_save = False
            elif k == ord('d'):
                print("Drawing")
                SeedlingDrawer.erase = False  # Activate drawing
                want_to_exit = False
                want_to_save = False
            elif k == ord('e'):
                print("Erasing")
                SeedlingDrawer.erase = True  # Activate erasing
                want_to_exit = False
                want_to_save = False
            elif k == ord('x'):
                if not want_to_exit:
                    print("If you REALLY want to exit, press 'x' again.")
                    want_to_exit = True
                else:
                    print("Exiting WITHOUT saving the image")
                    cv2.destroyAllWindows()
                    return None
            elif k == ord('s'):  # ESC key to exit
                if not want_to_save:
                    print("If you REALLY want to save, press 's' again.")
                    want_to_save = True
                else:
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
        self.OUTPUT_INPUT_PATH = os.path.join(self.OUTPUT_FOLDER, self.IN_SUFFIX)
        self.OUTPUT_GD_PATH = os.path.join(self.OUTPUT_FOLDER, self.OUT_SUFFIX)

    def make_directories(self):
        os.makedirs(os.path.join(self.OUTPUT_FOLDER), exist_ok=True)
        for folder in [self.IN_SUFFIX, self.OUT_SUFFIX]:
            os.makedirs(os.path.join(self.OUTPUT_FOLDER, folder), exist_ok=True)


    def already_done(self) -> Set[str]:
        """
        Return all the names with extension of the already done images

        Returns: 
        - `Set[str]`
        """
        return set(map(
            lambda path: os.path.split(path)[1],  
            glob.glob(os.path.join(self.OUTPUT_INPUT_PATH, '*'))
        ))

    def ground_truth_path_for(self, img_name) -> str:
        """
        Provides the full ground truth path for a img named `img_name`

        Parameters:
        - img_name: str as Path
            Name of the image with extension (e.g. .jpg,.png, ...)

        Returns:
        - str as Path
            Path to the desired ground truth image 
        """
        return os.path.join(self.OUTPUT_INPUT_PATH, img_name)

    def input_path_for(self, img_name) -> str:
        """
        Provides the full input path for a img named `img_name`

        Parameters:
        - img_name: str as Path
            Name of the image with extension (e.g. .jpg,.png, ...)

        Returns:
        - str as Path
            Path to the desired input image 
        """
        return os.path.join(self.OUTPUT_GD_PATH, img_name)

    def input_images_path(self):
        """
        Return the set of images paths that are on `INPUT_PATH`.

        Returns:
        - `Set[str]` of Paths
        """
        expandable_path = os.path.join(self.INPUT_FOLDER, '*')
        images_path = set(glob.glob(expandable_path))
        return images_path

    def todo_images(self) -> Iterable[str]:
        """
        Return the set of images paths that are willing to be done.

        Returns:
        - `[str]` of Paths
        """
        images_path = self.input_images_path()
        done_images = self.already_done()
        return filter(lambda path: os.path.split(path)[1] not in done_images,images_path)



    def run(self):
        self.make_directories()
        for img_path in self.todo_images():
            img_name = os.path.split(img_path)[1]

            io = SeedlingDrawer.image_drawer(img_path)

            if io is None:
                return

            in_img, out_img = io

            in_img_path = self.ground_truth_path_for(img_name)
            out_img_path = self.input_path_for(img_name)

            cv2.imwrite(in_img_path, in_img)
            cv2.imwrite(out_img_path, out_img)
