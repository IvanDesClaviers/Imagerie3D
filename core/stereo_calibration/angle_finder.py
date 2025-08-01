from core.utils import *
from core.stereo_calibration import *


def calibrate(cam_calib_path=CAMERA_CALIB_PATH, galv_calib_path=CAMERA_GALV_CALIB_PATH, verbose=0):
    """
    Compute the Camera stereo_calibration and the camera-Galvos stereo_calibration
    :param cam_calib_path: The path to the chessboard images taken from the camera
    :param galv_calib_path: The path to the chessboard with projected pattern images
     taken from the same camera
    :param verbose: Shows various information
    :return: The 3x3 camera intrinsic matrix
     The 5x1 camera distortion coefficient matrix
     The 4x4 Extrinsic Matrix between the camera and the galvos
     The 3x3 galvos intrinsic matrix
     The 5x1 galvos distortion coefficient matrix
    """
    if not os.path.exists(cam_calib_path):
        raise Exception("Camera Calibration images path Path incorrect ")
    if not os.path.exists(galv_calib_path):
        raise Exception("Camera Galvos Calibration images path Path incorrect ")

    cam_mtx, cam_dist = calibrate_camera(verbose, cam_calib_path)
    if verbose > 0:
        print("Camera Calibration Done")
        print()

    k_cam_galv, galv_mtx, galv_dist_coef = calibrate_galvos_to_camera(cam_mtx, cam_dist, verbose, galv_calib_path)
    if verbose > 0:
        print("Galvos-Camera Calibration Done")
        print()

    cv2.destroyAllWindows()
    return cam_mtx, cam_dist, k_cam_galv, galv_mtx, galv_dist_coef


def find_angle_grid(img_path: str, k_cam_galv=KEXT_GUESS, pattern_shape=LASER_PATTERN_SHAPE, cam_mtx=CAM_MTX,
                    cam_dist=CAM_DIST,
                    verbose=0):
    """
    Estimate the angle of the grid compared to a perpendicular perfect plan in the galvos coordinates.
    A laser pattern needs to be projected on it
    :param img_path: The path to the image of the grid with a laser pattern on it
    :param k_cam_galv: The 4x4 camera-galvos extrinsic matrix used
    :param pattern_shape: The projected pattern shape. It must be rectangular
    :param cam_mtx: The 3x3 camera intrinsic matrix
    :param cam_dist: The 5x1 camera distortion coef matrix
    :param verbose: Print various information based on its value
    :return: The tuple (grid normal[rad], point of the plan[mm]) in the galvos coordinates
    """
    im = cv2.imread(img_path)
    grid_img_points = find_laser_dots_on_calib_plan(im, 0, pattern_shape, 254)

    if verbose > 1:
        for c in grid_img_points:
            cv2.circle(im, tuple((int(c[0][0]), int(c[0][1]))), 1, (0, 0, 255), 2)
        show(im, "Laser Centers")
        cv2.destroyWindow("Laser Centers")

    print("Finding Grid Image points Done")
    print()

    # Create the grid plan coordinates like the chessboard centered at the grid center
    grid_galv_world_points, grid_normal = extract_grid_galvworld_pts(grid_img_points, k_cam_galv,
                                                                     generate_plan_coord(pattern_shape),
                                                                     cam_mtx, cam_dist, verbose=verbose)
    print("Finding Grid GalvosWorld points and normal Done")
    print()

    grid_plan = tuple((grid_normal, np.mean(grid_galv_world_points[:, 0, :].T)))

    perfect_plan = tuple((np.array((0, 0, 1)), np.array((0, 0, GRID_DISTANCE))))
    plot_vectors(perfect_plan[0], grid_plan[0], verbose=verbose)
    print("Plan Comparison finished successfully")

    return grid_plan
