import os

from core.stereo_calibration.config import *
from core.utils import compute_calib_error


def calibrate_camera(verbose=0, img_path=CAMERA_CALIB_PATH):
    """
    Calibrate the camera using pictures on the img_path folder
    :param verbose: Show the stereo_calibration results
    :param img_path: path to the folder
    :return:
    """
    # Set the chessboard plan, where z = 0
    obj_points = np.zeros((CHESSBOARD_SHAPE[1] * CHESSBOARD_SHAPE[0], 3), dtype='float32')
    obj_points[:, :2] = np.mgrid[0:CHESSBOARD_SHAPE[0], 0:CHESSBOARD_SHAPE[1]].T.reshape(-1, 2)
    obj_points = obj_points * SQUARE_SIZE

    chess_real_points = []  # 3d point in real world space
    chess_img_points = []  # 2d points in image plane.

    print(f"Loading pictures in '{img_path}' ...")
    # Iterate in our images
    gray = []
    for file in sorted(os.listdir(img_path)):
        img = cv2.imread(os.path.join(img_path, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SHAPE, cv2.CALIB_CB_FAST_CHECK)

        if ret:
            if verbose > 1:
                print(file)
            chess_real_points.append(obj_points)
            chess_img_points.append(corners)

    if len(chess_real_points) < 2:
        raise Exception("Chessboard not detected in enough pictures to calibrate. "
                        "Please use more input data.")
    print("Chess found in", len(chess_real_points), f"pictures / {len(os.listdir(img_path))}")
    print()

    ret, cam_mtx, cam_dist, cam_rot, cam_trans = cv2.calibrateCamera(chess_real_points, chess_img_points,
                                                                     gray.shape[::-1], None, CRITERIA)
    print("Initial Calibration Error: ", ret)
    print()
    if verbose > 0:
        print("Camera Intrinsic Parameters:", cam_mtx)
        print("Camera Distortion coef:", cam_dist)
        print()
    if verbose > 1:
        print("Computing re-projection with stored Values :")
        compute_calib_error(chess_real_points, chess_img_points, cam_mtx, cam_rot, cam_trans, cam_dist)
    if ret > 5:
        raise Exception("Pixel error is too high, the dataset must have problematic images. "
                        "Run with a higher verbose to find them, then remove them")

    return cam_mtx, cam_dist


if __name__ == "__main__":
    calibrate_camera(3)
