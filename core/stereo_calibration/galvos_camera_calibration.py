import os

from core.stereo_calibration.image_processing import create_galvos_virtual_coordinates, find_laser_dots_on_calib_plan
from core.stereo_calibration.maths_three_dim import get_laser_plan_3d_pts, get_camworld_point
from core.utils import *


def calibrate_galvos_to_camera(cam_mtx=CAM_MTX, cam_dist=CAM_DIST, verbose=0, galv_path=CAMERA_GALV_CALIB_PATH):
    """
    Main function performing the Galvos-camera stereo_calibration. This function read the camera intrinsic
    parameters and stereo_calibration parameters from the /utils/constants.py file and use them.

    For each image on the stereo_calibration dataset, the algorithm creates a galvos image, find the lasers dots
    and match them with 3D coordinates on the chessboard plan in order to compute the galvos-stereo_calibration and then
    the galvos-camera stereo_calibration.

    :return: The Galvos intrinsic parameters and the Galvos-Camera extrinsic parameters.
    """
    # All other function use the global variables, so I edit them here
    CAM_MTX = cam_mtx
    CAM_DIST = cam_dist

    # Initialize array pts with known values
    detected, chess_cam_img_pts, las_cam_img_pts, chess_plan_3d_pts, las_plan_3d_pts, las_galv_img_pts, \
        las_camworld_pts, chess_camworld_pts = initialize(verbose, galv_path)

    print("Laser detected in", len(detected), f"/{len(os.listdir(galv_path))} pictures")
    print()

    if len(detected) < 12:
        raise Exception("Chessboard not detected in enough pictures to calibrate. "
                        "Please use more input data.")

    # Find 3d coordinates of laser dots on the plane with rays method
    las_plan_3d_pts, plan_params = get_laser_plan_3d_pts(las_cam_img_pts, chess_cam_img_pts, chess_plan_3d_pts,
                                                         las_plan_3d_pts, detected, verbose)

    # Find galvos intrinsic matrix
    ret, galv_mtx, galv_dist_coef, galv_rot, galv_trans = cv2.calibrateCamera(las_plan_3d_pts, las_galv_img_pts,
                                                                              RESOLUTION, None, CRITERIA)

    # Get laser dots and chess corners CamWorld coordinates
    for k in range(len(detected)):
        for i in range(len(las_cam_img_pts[k])):
            las_camworld_pts[k][i] = get_camworld_point(las_cam_img_pts[k][i][0], plan_params[k])
        for j in range(len(chess_cam_img_pts[k])):
            chess_camworld_pts[k][j] = get_camworld_point(chess_cam_img_pts[k][j][0], plan_params[k])

    # Find our Extrinsic parameters
    cam_dist = CAM_DIST.copy()
    k_guess = KEXT_GUESS.copy()
    rms, _, _, _, galv_dist_coef, extrinsic_rot, extrinsic_tran, _, _, _ = \
        cv2.stereoCalibrateExtended(las_plan_3d_pts, las_cam_img_pts, las_galv_img_pts, CAM_MTX, cam_dist, galv_mtx,
                                    galv_dist_coef, RESOLUTION, k_guess[0:3, 0:3], np.array(k_guess[0:3, 3]),
                                    criteria=CRITERIA, flags=cv2.CALIB_USE_EXTRINSIC_GUESS | cv2.CALIB_FIX_INTRINSIC)

    # Since our galvos is not a camera, its intrinsic parameters is actually the inverse of what we found
    galv_mtx = np.linalg.inv(galv_mtx)

    # The  of stereocalib is the translation rotation to have coordinates from galvos to camera. We want the inverse
    k_cam_galv = np.linalg.inv(create_extrinsic(extrinsic_rot, extrinsic_tran))

    print(f"Initial Galvos stereo_calibration Error = {ret} pixels")
    print(f"Initial Camera Galvos Stereo stereo_calibration Error= {rms} pixels")
    print()
    if verbose > 1:
        print("Galvos Intrinsic Parameters:", galv_mtx)
        print("Galvos \"Camera\" Intrinsic Parameters:", np.linalg.inv(galv_mtx))
        print()
        print(f"Galvos-Camera Extrinsic Parameters: {k_cam_galv}")
        print(f"Galvos-Camera Extrinsic Rotation (deg): {cv2.Rodrigues(extrinsic_rot)[0] * 180 / pi}")
        print()
    if verbose > 2:
        print(f"Computing galvos stereo_calibration reprojection error...")
        compute_calib_error(las_plan_3d_pts, las_galv_img_pts, np.linalg.inv(galv_mtx), galv_rot, galv_trans,
                            galv_dist_coef)
        print()
    compare_galvos_extrinsic(k_cam_galv)
    if ret > 5 or rms > 5:
        raise Exception("Pixel error is too high, the dataset must have problematic images. "
                        "Run with a higher verbose to find them")

    return k_cam_galv, galv_mtx, galv_dist_coef


def initialize(verbose: int, galv_path: str):
    """
    Initialize all arrays with the correct shape for the opencv stereo_calibration functions.
    :return: All initialized arrays.
    """
    # Create the chessboard plan, where all points have z = 0
    objp = np.zeros((CHESSBOARD_SHAPE[1] * CHESSBOARD_SHAPE[0], 3), dtype='float32')
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SHAPE[0], 0:CHESSBOARD_SHAPE[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE

    chess_cam_img_pts = []  # Chessboard pts in 2D (pixels) relative to camera
    las_cam_img_pts = []  # Laser pts in 2D (pixels) relative to camera
    las_galv_img_pts = []  # Laser pts in 2D relative to galvos

    las_camworld_pts = []  # 3D Coordinates of the laser pts in real world in CAMERA REFERENTIAL
    chess_camworld_pts = []  # 3D Coordinates of the chessboard pts in real world in CAMERA REFERENTIAL

    chess_plan_3d_pts = []  # chessboard pts in 3D relative to the plane in CAMERA REFERENTIAL
    las_plan_3d_pts = []  # laser pts in 3D relative to the plane in CAMERA REFERENTIAL for each image

    # Find chessboard AND laser pts in the image
    detected = []  # detected files
    print("Reading Galvos-camera stereo_calibration dataset")
    print()
    for file in sorted(os.listdir(galv_path)):
        name = os.path.join(galv_path, file)
        img = cv2.imread(name)
        img = cv2.resize(img, (RESOLUTION[1], RESOLUTION[0]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SHAPE, cv2.CALIB_CB_FAST_CHECK)
        if ret:
            detected.append(os.path.join(galv_path, file))
            if verbose > 1:
                print(file)

            chess_cam_img_pts.append(corners)  # Get chessboard image points coordinates
            chess_plan_3d_pts.append(objp)  # Associate the chessboard plan coordinates

            las_cam_img_pts.append(find_laser_dots_on_calib_plan(img, verbose))  # Get image coordinates of laser dots
            las_galv_img_pts.append(create_galvos_virtual_coordinates(img, 0))  # Create galvos image coordinates

            # Initialize other array according to the number of points detected
            las_plan_3d_pts.append(np.zeros([las_cam_img_pts[len(las_cam_img_pts) - 1].shape[0], 3], dtype=np.float32))
            las_camworld_pts.append(np.zeros([las_cam_img_pts[len(las_cam_img_pts) - 1].shape[0], 3], dtype=np.float32))
            chess_camworld_pts.append(
                np.zeros([chess_cam_img_pts[len(chess_cam_img_pts) - 1].shape[0], 3], dtype=np.float32))

    return detected, chess_cam_img_pts, las_cam_img_pts, chess_plan_3d_pts, las_plan_3d_pts, las_galv_img_pts, \
        las_camworld_pts, chess_camworld_pts
