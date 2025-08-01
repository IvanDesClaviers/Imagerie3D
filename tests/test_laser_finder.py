import os
import unittest

from core import GroundTruthManager, CamType
from core.utils import *
from core.galvos_calibration.grid_processing import find_laser_accurately, find_laser


class TestLaserFinder(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLaserFinder, self).__init__(*args, **kwargs)

    def test_find_laser_accurately(self):
        for filename in os.listdir("test_images/laser"):
            image = cv2.imread(os.path.join("test_images/laser", filename))
            find_laser_accurately(image, verbose=2)

    def test_find_laser_on_ids(self):
        from core.cameras import IdsCamera
        cam = IdsCamera()
        cam.start_video()
        frame = cam.get_frame()

        cam_mask = np.zeros(frame.shape[0:2], dtype=np.uint8)
        cv2.fillPoly(cam_mask, [np.array(GroundTruthManager.load_quadrant_mask(CamType.TOP_RIGHT))], 255)
        while True:
            find_laser(frame, cam_mask, verbose=5)
            frame = cam.get_frame()
            if cv2.waitKey(1) > -1:
                break
        cam.stop_video()
