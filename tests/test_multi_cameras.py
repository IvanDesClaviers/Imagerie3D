import unittest

from core.cameras import CamType
from core.galvos_calibration import MultipleCameras


class TestMultipleCameras(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMultipleCameras, self).__init__(*args, **kwargs)

    def test_mult_ids_camera_one_by_one(self):
        cams = MultipleCameras()
        cams.test_cameras_one_by_one()

    def test_snapshot_all_cams(self):
        cams = MultipleCameras()
        cams.save()

    def test_snapshot_one_cams(self):
        cams = MultipleCameras()
        cams.save(CamType.TOP_LEFT, "Laser_TopLeft_dark_dim_out2.png")
