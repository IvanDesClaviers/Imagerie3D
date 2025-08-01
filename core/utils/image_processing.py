import cv2
import numpy as np

from core import ComputerVisionException
from core.utils import get_intrinsic_from_datasheet, show


def undistort_grid_img(frame: np.array, show_img=False):
    """

    :param frame:
    :param show_img:
    :return:
    """
    cam_mtx = get_intrinsic_from_datasheet(frame.shape, (8.5 * 10 ** -3, 8.5 * 10 ** -3),
                                           (6.1410 ** -3, 4.60510 ** -3))
    coeffs = np.array([[-0.0], [-0.0], [0.0], [0.0], [0.0]])
    undistorded = cv2.undistort(frame, cam_mtx, coeffs)
    if show_img:
        show(undistorded, "Undist", block=False)

    return undistorded


def equalize_image(img: np.array):
    """

    :param img:
    :return:
    """
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def create_losange_morph(size: tuple = (5, 5)):
    """

    :param size:
    :return:
    """
    if size[0] != size[1]:
        raise ComputerVisionException("Rectangular losange not supported")
    n = size[0]
    losange = []
    for k in range(n // 2):
        vector = np.ones(n, dtype=np.uint8)
        vector[0:n // 2 - k] = 0
        vector[n // 2 + 1 + k:n] = 0
        losange.append(vector)
    losange.append(np.ones(n, dtype=np.uint8))
    vector = losange.copy()
    for k in range(2, len(losange) + 1):
        losange.append(vector[-k])
    return np.array(losange, dtype=np.uint8)


def inverse_matrix_first_channels(mat: np.array):
    new_mat = np.zeros(mat.shape)
    for i in range(new_mat.shape[0]):
        for j in range(new_mat.shape[1]):
            new_mat[i, j, 0] = mat[j, i, 1]
            new_mat[i, j, 1] = mat[j, i, 0]
    return new_mat
