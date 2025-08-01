import os
import time

import cv2

from core.utils.data_shower import show
from global_config import IMG_DIR


class Camera:
    id: int = -1  # The camera ID, -1 means it has not been initialized
    # NO_ROTATION = -1
    # ROTATE_90_CLOCKWISE = 0
    # ROTATE_180 = 1
    # ROTATE_90_COUNTERCLOCKWISE = 2
    rotation: int = -1
    serial_num: str = None  # Some camera may have a serial number

    def __init__(self, rot=-1):
        self.rotation = rot

    def start_video(self):
        """
        This method start the video stream if it was not already open.
        It will do nothing if the video stream was open
        """
        pass

    def stop_video(self):
        """ Stop the video stream """
        pass

    def get_frame(self):
        """ Get the current frame of the camera"""
        pass

    def is_open(self):
        """ Check if the camera has started and can have its frames captured """
        pass

    def close_camera(self):
        """Definitely close the camera"""
        pass

    def show_camera_stream(self):
        """ Start and show the video stream. It closes by pressing any key"""
        self.start_video()
        while True:
            frame = self.get_frame()
            if cv2.waitKey(1) > -1:
                break
            show(frame, "Camera Stream", block=False)
        self.stop_video()
        cv2.destroyAllWindows()

    def save(self, name="image.png"):
        """Save the image of the camera after opening it"""
        self.start_video()
        time.sleep(1)
        frame = self.get_frame()
        time.sleep(1)
        cv2.imwrite(os.path.join(IMG_DIR, name), frame)
        self.stop_video()
