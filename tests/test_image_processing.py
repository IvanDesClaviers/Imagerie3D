import unittest

from core import GroundTruthManager, CamType
from core.galvos_calibration.grid_processing import *


class TestImageProcessing(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestImageProcessing, self).__init__(*args, **kwargs)

    def test_quadrant_detection(self):
        for filename in os.listdir(IMG_DIR):
            if "CamType.TOP_RIGHT_2022.3.1-14h4.png" in filename:
                print(f"Reading {filename}")
                frame = cv2.imread(os.path.join(IMG_DIR, filename))
                show(frame, "original")
                intersection_img, intersections = get_all_grid_intersections(frame, show_img=True)
                quadrant, quadrant_intersections = delimit_quadrant(intersection_img, intersections)
                get_calibration_intersections(quadrant_intersections, 41, frame.shape, show_img=True)

    def test_pt_sort(self):
        for filename in os.listdir(IMG_DIR):
            if "CamType.TOP_RIGHT_2022.2.24-13h29_quadrant.png" in filename:
                frame = cv2.imread(os.path.join(IMG_DIR, filename))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                quadrant_intersect = get_all_center_coord(gray)
                res = order_point_custom(quadrant_intersect, frame_shape=frame.shape)
                final = interpolate_missing_intersections(res, 41)

                show_grid_intersections(final, frame.shape)

    def test_laser_detection(self):
        for filename in os.listdir(IMG_DIR):
            if "Laser_" in filename:
                try:
                    frame = cv2.imread(os.path.join(IMG_DIR, filename))

                    mask = np.zeros(frame.shape[0:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [np.array(GroundTruthManager.load_quadrant_mask(CamType.TOP_LEFT))], 255)

                    inter = find_laser(frame, mask, 5)
                    inter2 = find_laser_accurately(frame, mask, 4)
                    print(inter)
                    print(inter2)
                    print("")
                    cv2.waitKey()

                except ComputerVisionException:
                    print(filename)
                    print("Failed")
