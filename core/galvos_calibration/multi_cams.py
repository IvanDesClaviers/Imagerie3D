import time
import cv2

from core.utils.bcolors import bcolors
from core.galvos_calibration.config import *
from core.exceptions import CameraException, ComputerVisionException
from core.cameras import Camera, OpencvCamera, IdsCamera, CamType


class MultipleCameras:
    # The cameras
    cam_dict: dict = {CamType.TOP_RIGHT: None,
                      CamType.TOP_LEFT: None,
                      CamType.BOT_RIGHT: None,
                      CamType.BOT_LEFT: None}

    def __init__(self, w=RESOLUTION[1], h=RESOLUTION[0], cam_mode: int = 1, rot_dict=None):
        """
        Create the Quad cam class. All camera are of the same type. 
        :param w: The width in pixel of the camera, default value is from the config file.
        :param h: The width in pixel of the camera, default value is from the config file.
        :param cam_mode: The type of camera we will use:
            - 1 for the ids camera (default)
            - 2 for opencv camera (testing purpose)
            - 3 for the abstract camera (testing purpose)
        :param rot_dict: Depending on the setup, some camera may be rotated.
        The default rotation are hardcoded.
        """
        # Each camera has a specific rotation depending on their type
        if rot_dict is None:
            rot_dict = {CamType.TOP_RIGHT: cv2.ROTATE_90_CLOCKWISE,
                        CamType.TOP_LEFT: -1,
                        CamType.BOT_RIGHT: -1,
                        CamType.BOT_LEFT: cv2.ROTATE_90_CLOCKWISE}
        for cam_type in self.cam_dict.keys():
            try:
                if cam_mode == 1:
                    cam = IdsCamera(w, h)
                elif cam_mode == 2:  # We just have one camera
                    cam = OpencvCamera()
                    self.cam_dict = {CamType.TOP_RIGHT: cam,
                                     CamType.TOP_LEFT: cam,
                                     CamType.BOT_RIGHT: cam,
                                     CamType.BOT_LEFT: cam}
                    return
                else:
                    cam = Camera()
            except CameraException:
                print("Error During Ids Camera setup\n")
                print("Reason:\n")
                print("Make sure all cameras are connected and have a green light in front of them\n")
                return
            if cam.serial_num is not None:
                cam_type = SERIAL_NUMBER_ROLE[cam.serial_num]
                self.cam_dict[cam_type] = cam
                self.cam_dict[cam_type].rotation = rot_dict[cam_type]
            else:  # Randomly define them
                self.cam_dict[cam_type] = cam

    def _define_camera_role(self, cam: Camera):
        """
        Show the video stream of the camera, and ask the user to select its position among the available ones
        :param cam: The camera
        """
        print("Showing camera Stream, click on the picture to continue...\n")
        cam.show_camera_stream()
        for cam_name in self.cam_dict.keys():
            if self.cam_dict[cam_name] is None:
                print(f"{cam_name} : write '{cam_name.value}'")
                if list(self.cam_dict.values()).count(None) == 1:
                    print("Big brain time: last camera to initialize is this one :)\n")
                    self.cam_dict[cam_name] = cam
                    return
        print("Which camera was it? ")
        x = input()
        proceed = False
        while not proceed:
            try:
                if CamType(x) in self.cam_dict.keys():
                    if self.cam_dict[CamType(x)] is None:
                        proceed = True
                        self.cam_dict[CamType(x)] = cam
                    else:
                        print("This camera has already been initialized")
                        x = input()
                else:
                    print(f"The input must be among the camera tag listed above")
                    x = input()
            except ValueError:
                print(f"The input must be among the camera tag listed above")
                x = input()
        print()

    @staticmethod
    def chose_new_center(redefine_center: bool, default_camera_type=CamType.BOT_RIGHT):
        """ Chose which camera holds the center.
        :param redefine_center: Return the default_camera_type if false
        :param default_camera_type: Set to the Bottom Left camera (arbitrary choice).
        :return: The camera type with the center
        """
        cam_with_center = default_camera_type
        if redefine_center:
            proceed = False
            print("Do you want to use this camera for the grid center point?")
            x = input()
            while not proceed:
                try:
                    cam_with_center = CamType(x)
                except ValueError:
                    print(f"The input must be among the camera tag listed above")
                    x = input()
        return cam_with_center

    def test_cameras_one_by_one(self):
        """ Show all cameras one by one """
        try:
            for k in range(len(self.cam_dict.keys())):
                cam_name = list(self.cam_dict.keys())[k]
                print(f"{bcolors.WARNING}Showing {cam_name} camera ...")
                self.cam_dict[cam_name].show_camera_stream()
        except (ComputerVisionException, CameraException):
            print("Error while configuring cameras")
        print()

    def get_cam_list(self):
        """ Get all cameras """
        return list(self.cam_dict.values())

    def open_all(self):
        """ Open all cameras """
        for cam in self.cam_dict.values():
            cam.start_video()

    def close_all(self):
        """ Close all cameras """
        for cam in self.cam_dict.values():
            cam.close_camera()

    def save(self, cam_type: CamType = None, name: str = None):
        """
        Save the current frame of the cameras. If no camera type is specified,
        take a snapshot of all available cameras.
        :param cam_type: The specific camera we want to save a picture of
        :param name: The name of the file. It is by default the timestamp of the snapshot
        """
        date = time.localtime()
        if cam_type is None:
            for camType in self.cam_dict.keys():
                self.cam_dict[camType].save(
                    f"{camType}_{date.tm_year}.{date.tm_mon}.{date.tm_mday}-{date.tm_hour}h{date.tm_min}.png")
        else:
            if name is None:
                name = f"{cam_type}_{date.tm_year}.{date.tm_mon}.{date.tm_mday}-{date.tm_hour}h{date.tm_min}.png"
            self.cam_dict[cam_type].save(name)
