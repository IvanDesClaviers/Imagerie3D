import unittest

from core.cameras import *


class TestCameras(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCameras, self).__init__(*args, **kwargs)

    def test_opencv_camera(self):
        cam = OpencvCamera()
        cam.start_video()
        while True:
            frame = cam.get_frame()
            # ...and finally display it
            name = "SimpleLive_Python_uEye_OpenCV"
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, 1280, 720)
            cv2.imshow(name, frame)
            if cv2.waitKey(1) > -1:
                break

        cam.stop_video()

    def test_ids_camera(self):
        cam = IdsCamera()
        cam.start_video()
        while True:
            frame = cam.get_frame()
            if cv2.waitKey(1) > -1:
                break
            cv2.namedWindow("a", cv2.WINDOW_NORMAL)
            cv2.imshow("a", frame)
        cam.stop_video()

    def test_calibrate_camera_with_rangefinder(self):
        cam = IdsCamera()
        cam.calibrate_camera_with_rangefinder()

    def test_laser_radius(self):
        cam = IdsCamera()
        cam.start_video()
        while True:
            frame = cam.get_frame()
            name = "SimpleLive_Python_uEye_OpenCV"
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, 1280, 720)
            cv2.imshow(name, frame)
            if cv2.waitKey(1) > -1:
                cv2.imwrite("test_images/laser/Laser_precise_center2.png", frame)
                im = frame[int(2 / 8 * frame.shape[0]):int(6 / 8 * frame.shape[0]),
                           int(2 / 8 * frame.shape[1]):int(6 / 8 * frame.shape[1]), 1]
                show(im, f"Orig", block=False)
                for i in range(200, 256, 5):
                    _, im_i = cv2.threshold(im, i, 255, cv2.THRESH_BINARY)
                    show(im_i, f"Bin{i}", block=False)
                cv2.waitKey()
                break

        cam.stop_video()
