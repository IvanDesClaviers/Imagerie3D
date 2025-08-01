import unittest

from core.galvos_calibration import GalvosCalibrator


class TestGalvosCalib(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestGalvosCalib, self).__init__(*args, **kwargs)

    def test_calibrate(self):
        galv_calib = GalvosCalibrator()
        galv_calib.calibrate()

    def test_calibrate_fast(self):
        galv_calib = GalvosCalibrator(quick=True)
        galv_calib.calibrate()

    def test_calibrate_dummy(self):
        galv_calib = GalvosCalibrator(dummy=True, quick=True, save=False)
        galv_calib.calibrate()
