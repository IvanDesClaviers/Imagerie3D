import unittest
import numpy.testing as npt

from core.stereo_calibration import *
from global_config import *


class TestStereoCalib(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestStereoCalib, self).__init__(*args, **kwargs)
        self.test_path = "test_images"
        self.angle_image_path = "test_images/virtual_chess"
        self.cam_calib_path = os.path.join(ROOT_DIR, "core/stereo_calibration/calibration_images/chessboard")
        self.galv_calib_path = os.path.join(ROOT_DIR, "core/stereo_calibration/calibration_images/galvos")
        self.test_grid_path = "test_images/grid/raw_grid.bmp"

    def test_calibration_bad_dir(self):
        with self.assertRaises(Exception) as e:
            assert calibrate("", "")
        print(e.exception)
        with self.assertRaises(Exception) as e:
            calibrate(self.cam_calib_path, "")
        print(e.exception)
        with self.assertRaises(Exception) as e:
            calibrate("test_images", self.galv_calib_path)
        print(e.exception)
        with self.assertRaises(Exception) as e:
            calibrate(self.angle_image_path, self.galv_calib_path)
        print(e.exception)

        with self.assertRaises(Exception) as e:
            calibrate_galvos_to_camera(galv_path="test_images")
        print(e.exception)
        with self.assertRaises(Exception) as e:
            calibrate_galvos_to_camera(galv_path="test_images/threshed_grid")
        print(e.exception)
        with self.assertRaises(Exception) as e:
            calibrate_galvos_to_camera(galv_path=self.angle_image_path)
        print(e.exception)

    def test_calibration_bad_extrinsic(self):
        with self.assertRaises(Exception) as e:
            calibrate_galvos_to_camera(galv_path="test_images/test_bad_extrins", verbose=3)
        print(e.exception)

    def test_bad_laser_angle_finder(self):
        with self.assertRaises(Exception) as e:
            find_angle_grid(os.path.join(self.cam_calib_path, "1.bmp"), verbose=0)
        print(e.exception)

    def test_working_calibration(self):
        cam_mtx, camdist, k_cam_galv, _, _ = calibrate(self.cam_calib_path, self.galv_calib_path, 0)
        npt.assert_almost_equal(k_cam_galv, KEXT_GUESS, 1)
        npt.assert_almost_equal(cam_mtx, CAM_MTX, 1)
        npt.assert_almost_equal(camdist, CAM_DIST, 1)

    def test_working_find_grid_angle(self):
        find_angle_grid(self.test_grid_path, verbose=0)

    def test_pnp_ransac_accuracy(self):
        objp = np.zeros((CHESSBOARD_SHAPE[1] * CHESSBOARD_SHAPE[0], 3), dtype='float32')
        objp[:, :2] = np.mgrid[0:CHESSBOARD_SHAPE[0], 0:CHESSBOARD_SHAPE[1]].T.reshape(-1, 2)
        objp = objp * SQUARE_SIZE

        for file in sorted(os.listdir(self.angle_image_path)):
            name = os.path.join(self.angle_image_path, file)
            img = cv2.imread(name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SHAPE)
            if ret:
                print(file)
                # Associate the chessboard plan coordinates
                cam_photoshop = np.array([[4.51652637e+03, 0.00000000e+00, 1.54304577e+03],
                                          [0.00000000e+00, 4.30140221e+03, 1.04039916e+03],
                                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
                ret, rvec, p, inliers = cv2.solvePnPRansac(objp, corners,
                                                           cam_photoshop,
                                                           np.array([[0, 0, 0, 0, 0]], dtype=np.float32))
                print("Angle found by ransac:", rvec[1] * 180 / 3.14)
                print()
