import os
import time

import cv2
import numpy as np
from pyueye import ueye

from core.utils import bcolors
from core.cameras import Camera
from core.exceptions import CameraException
from global_config import IMG_DIR


class IdsCamera(Camera):
    _open: bool = False  # Is the initialization a success

    def __init__(self, w=4912, h=3684, rotation=-1):
        super().__init__(rotation)
        # Variables
        self._pcImageMemory = ueye.c_mem_p()
        self._MemID = ueye.int()
        self._pitch = ueye.INT()
        self._nBitsPerPixel = ueye.INT(24)  # 24: bits per pixel for color mode; take 8 bits per pixel for monochrome
        self._bytes_per_pixel = int(self._nBitsPerPixel / 8)
        self._width = ueye.INT(w)
        self._height = ueye.INT(h)

        try:
            self._setup()
        except CameraException:
            print(f"{bcolors.FAIL}Camera initialization failed\n")

    def _setup(self):
        print(f"Starting configuration of ids camera...")
        array = [1, 2, 3]
        n_range = (ueye.c_uint * len(array))(*array)
        new_framerate = ueye.c_double()
        new_expo = ueye.c_double()
        enable = ueye.c_double()
        zero = ueye.c_double(0)
        n_gamma = ueye.c_int(160)
        s_info = ueye.SENSORINFO()
        c_info = ueye.CAMINFO()
        _id = ueye.HIDS(0)  # Takes first available camera
        # Starts the driver and establishes the connection to the camera
        try:
            if ueye.is_InitCamera(_id, None) != ueye.IS_SUCCESS:
                raise CameraException("is_InitCamera ERROR")
        except OSError:
            raise CameraException("For unknown reasons, camera image aquisition is corrupted. "
                                  "Un-plug/ plug the camera cable at both ends and retry")
        self.id = int(_id)

        # Reads out the data hard-coded in the non-volatile camera memory and writes
        #  it to the data structure that cInfo points to
        if ueye.is_GetCameraInfo(ueye.HIDS(self.id), c_info) != ueye.IS_SUCCESS:
            raise CameraException("is_GetCameraInfo ERROR")

        print("Camera info:")
        print(f"Serial Number: {c_info.SerNo}")
        print(f"\tManufacturer: {c_info.ID}")
        print(f"\tControl Date: {c_info.Date}")
        self.serial_num = c_info.SerNo

        # You can query additional information about the sensor type used in the camera
        if ueye.is_GetSensorInfo(ueye.HIDS(self.id), s_info) != ueye.IS_SUCCESS:
            raise CameraException("is_GetSensorInfo ERROR")

        print(f"\tCamera Model: {s_info.strSensorName}")
        print(f"\tWidth = {s_info.nMaxWidth}")
        print(f"\tHeight = {s_info.nMaxHeight}")
        print(f"\tPixel Size: {(s_info.wPixelSize / 100.0)} μm")

        if ueye.is_PixelClock(ueye.HIDS(self.id), ueye.IS_PIXELCLOCK_CMD_GET_RANGE, n_range,
                              ueye.sizeof(n_range)) != ueye.IS_SUCCESS:
            raise CameraException("IS_PIXELCLOCK_CMD_GET_RANGE ERROR")
        if ueye.is_PixelClock(ueye.HIDS(self.id), ueye.IS_PIXELCLOCK_CMD_SET, n_range[1],
                              ueye.sizeof(n_range[1])) != ueye.IS_SUCCESS:
            raise CameraException("IS_PIXELCLOCK_CMD_SET ERROR")

        print(f"\tPixel Clock = {n_range[1]} MHz")

        if ueye.is_SetFrameRate(ueye.HIDS(self.id), 20.0, new_framerate) != ueye.IS_SUCCESS:
            raise CameraException("is_SetFrameRate ERROR")

        print(f"\tFrame Rate = {new_framerate}")

        if ueye.is_Exposure(ueye.HIDS(self.id), ueye.IS_EXPOSURE_CMD_GET_LONG_EXPOSURE_RANGE_MAX, new_expo,
                            8) != ueye.IS_SUCCESS:
            raise CameraException("IS_EXPOSURE_CMD_GET_LONG_EXPOSURE_RANGE_MAX ERROR")
        if ueye.is_Exposure(ueye.HIDS(self.id), ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, new_expo, 8) != ueye.IS_SUCCESS:
            raise CameraException("IS_EXPOSURE_CMD_SET_EXPOSURE ERROR")

        print(f"\tExposure = {new_expo}")

        if ueye.is_SetAutoParameter(ueye.HIDS(self.id), ueye.IS_SET_ENABLE_AUTO_WHITEBALANCE, enable,
                                    zero) != ueye.IS_SUCCESS:
            raise CameraException("IS_SET_ENABLE_AUTO_WHITEBALANCE ERROR")
        if ueye.is_SetAutoParameter(ueye.HIDS(self.id), ueye.IS_SET_ENABLE_AUTO_GAIN, enable, zero) != ueye.IS_SUCCESS:
            raise CameraException("IS_SET_ENABLE_AUTO_GAIN ERROR")
        if ueye.is_Gamma(ueye.HIDS(self.id), ueye.IS_GAMMA_CMD_SET, n_gamma, ueye.sizeof(n_gamma)) != ueye.IS_SUCCESS:
            raise CameraException("IS_GAMMA_CMD_SET ERROR")

        print(f"\tGamma = {n_gamma}")

        # Allocates an image memory for an image having its dimensions defined by width
        #  and height and its color depth defined by nBitsPerPixel
        if ueye.is_AllocImageMem(ueye.HIDS(self.id), self._width, self._height, self._nBitsPerPixel,
                                 self._pcImageMemory,
                                 self._MemID) != ueye.IS_SUCCESS:
            raise CameraException("is_AllocImageMem ERROR")

        if ueye.is_SetImageMem(ueye.HIDS(self.id), self._pcImageMemory, self._MemID) != ueye.IS_SUCCESS:
            raise CameraException("is_SetImageMem ERROR")

        if ueye.is_SetDisplayMode(ueye.HIDS(self.id), ueye.IS_SET_DM_DIB) != ueye.IS_SUCCESS:
            raise CameraException("is_SetDisplayMode ERROR")

        self._open = True
        print(f"{bcolors.OKGREEN}Configuration of ids camera number {self.id} complete!")
        print()

    def start_video(self):
        try:
            if not self._open:
                raise CameraException("Setup the camera before trying to start it")

            if ueye.is_CaptureVideo(ueye.HIDS(self.id), ueye.IS_DONT_WAIT) != ueye.IS_SUCCESS:
                raise CameraException("is_CaptureVideo ERROR")
            # Enables the queue mode for existing image memory sequences
            if ueye.is_InquireImageMem(ueye.HIDS(self.id), self._pcImageMemory, self._MemID, self._width, self._height,
                                       self._nBitsPerPixel, self._pitch) != ueye.IS_SUCCESS:
                raise CameraException("is_InquireImageMem ERROR")

        except CameraException as err:
            print(f"{bcolors.FAIL}{err}\n")

    def get_frame(self):
        try:
            if not self._open:
                raise CameraException("You should open the camera stream before capturing a frame")

            # In order to display the image in an OpenCV window we need to...
            # ...extract the data of our image memory
            array = ueye.get_data(self._pcImageMemory, self._width, self._height, self._nBitsPerPixel, self._pitch,
                                  copy=False)

            # ...reshape it in an numpy array...
            array = np.reshape(array, (self._height.value, self._width.value, self._bytes_per_pixel))
            if self.rotation != -1:
                array = cv2.rotate(array, self.rotation)
            return array
        except CameraException as err:
            print(f"{bcolors.FAIL}Reason: {err}")
            return np.zeros((self._width.value, self._height.value))

    def stop_video(self):
        try:
            if not self._open:
                raise CameraException("Can't stop a stream not opened")

            if ueye.is_StopLiveVideo(ueye.HIDS(self.id), ueye.IS_WAIT) != ueye.IS_SUCCESS:
                raise CameraException("is_StopLiveVideo ERROR")
        except CameraException as err:
            print("Video stream failed")
            print(f"Reason: {err}")

    def close_camera(self):
        # Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
        ueye.is_FreeImageMem(ueye.HIDS(self.id), self._pcImageMemory, self._MemID)

        # Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
        ueye.is_ExitCamera(ueye.HIDS(self.id))
        self._open = False
        print(f"Camera {self.id} closed successfully")

    def is_open(self):
        return self._open

    def calibrate_camera_with_rangefinder(self):
        self.start_video()
        nb_zoom = 1
        max_z = 5
        print("Press 'z' to zoom in and 'x' to zoom out")
        while True:
            frame = self.get_frame()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            key = cv2.waitKey(1)
            if key == ord('z'):
                if nb_zoom < max_z:
                    nb_zoom += 1
            elif key == ord('x'):
                if 1 < nb_zoom:
                    nb_zoom -= 1
            elif key > -1:
                break

            frame = cv2.circle(frame, (self._width // 2, self._height // 2), nb_zoom, (0, 0, 255), 3)
            frame = cv2.circle(frame, (self._width // 2, self._height // 2), 10 * nb_zoom, (0, 0, 255), 3)
            cv2.namedWindow("a", cv2.WINDOW_NORMAL)
            cv2.imshow("a", frame)
            cv2.resizeWindow("a", 640, 480)

        self.stop_video()

    def save(self, name="image.png"):
        print(f"Saving {os.path.join(IMG_DIR, name)} ...")
        self.start_video()
        time.sleep(1)
        frame = self.get_frame()
        time.sleep(1)
        cv2.imwrite(os.path.join(IMG_DIR, name), frame)
        self.stop_video()
