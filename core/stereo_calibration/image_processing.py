from core.stereo_calibration.config import *
from core.utils import order_points


def create_galvos_virtual_coordinates(img, verbose=0):
    """
    Creates the coordinates of the laser dots if the galvos was acquiring an image
    :param img: The galvos-camera stereo_calibration processed image
    :param verbose: Show the virtual image if 2 or more
    :return: An array with the theoretical centers of the laser dots projected by the galvos
    """
    galv_coord = np.zeros((LASER_PATTERN_SHAPE[0] * LASER_PATTERN_SHAPE[1], 1, 2), dtype=np.float32)
    height, width = img.shape[0:2]
    # 7 de 20
    offx = int(PROJECTION_OFFSET[0] / GRID_SIZE[0] * width)
    offy = (PROJECTION_OFFSET[1] / GRID_SIZE[1] * height)
    x_step = (width - 2 * offx) / (LASER_PATTERN_SHAPE[0] - 1)
    y_step = (height - 2 * offy) / (LASER_PATTERN_SHAPE[1] - 1)

    # Order of points is left to right, top to bottom
    for j in range(LASER_PATTERN_SHAPE[1]):
        for i in range(LASER_PATTERN_SHAPE[0]):
            galv_coord[i + j * LASER_PATTERN_SHAPE[0]] = np.array([int(offx + i * x_step),
                                                                   int(offy + j * y_step)])
    if verbose > 1:
        im = np.zeros(img.shape, dtype=np.float32)
        for coord in galv_coord:
            cv2.circle(im, (int(coord[0][0]), int(coord[0][1])), 5, (0, 200, 0), 3)

    return galv_coord


def find_laser_dots_on_calib_plan(img, verbose=0, pattern=LASER_PATTERN_SHAPE, thresh=240):
    """
    Find the center of the laser dots in an image and return them ordered from top to bottom, left to right
    WARNING: The function is unstable if the environment is too bright.
    Make sure to Take pictures in a low light environment
    :return: The (pattern[0]*pattern[1])x1x2 points pixel coordinates in the original image referential
    """
    green = img[:, :, 1]
    ret, thresh = cv2.threshold(green, thresh, 255, cv2.THRESH_BINARY)
    laser_contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    laser_contours = sorted(laser_contours, key=len)

    while len(laser_contours) > pattern[0] * pattern[1]:
        len_list = [len(c) for c in laser_contours]
        mean_length = len_list[len(len_list) // 2]
        if abs(len_list[0] - mean_length) > abs(len_list[-1] - mean_length):
            laser_contours = laser_contours[1:]
        else:
            laser_contours = laser_contours[0:len(laser_contours) - 1]

    if len(laser_contours) != pattern[0] * pattern[1]:
        raise Exception("Galvos laser dot detection failed, check input images.")

    # Evaluate laser contours centers
    laser_centers = np.zeros((len(laser_contours), 1, 2), dtype="float32")
    for k in range(pattern[0] * pattern[1]):
        c = laser_contours[k]
        m = cv2.moments(c)
        if m['m00'] != 0.0:
            laser_centers[k] = [m['m10'] / m['m00'], m['m01'] / m['m00']]
        else:
            br = cv2.boundingRect(c)
            laser_centers[k] = [br[0] + br[2] / 2, br[1] + br[3] / 2]

    return order_points(img, laser_centers, verbose)

