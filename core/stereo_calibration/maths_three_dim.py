from core.stereo_calibration.config import *
from core.utils import create_extrinsic


def get_plan_param(cam_img_pts, plan_3d_pts, cam_mtx=CAM_MTX, cam_dist=CAM_DIST):
    """
    Extract a plan extrinsic parameters using pnpRansac along with the parameters for its equation
    :param cam_img_pts: pixel 2d Coordinates of a plan
    :param plan_3d_pts: 3d Coordinates of a plan in mm
    :param cam_mtx: The 3x3 camera intrinsic matrix
    :param cam_dist: The 5x1 camera distortion coef matrix
    :return: (kext, d, normal) tuple, kext being a 4x4 matrix of external parameters and
    normal the normal=[a,b,c] to the plan, and d being the constant of the plan equation
    ax + by + cz + d = 0
    """
    # Find extr of camera/ plane for each image
    ret, rvec, p, inliers = cv2.solvePnPRansac(plan_3d_pts, cam_img_pts, cam_mtx, cam_dist)
    r_plane = cv2.Rodrigues(rvec)[0]  # We want a 3x3 rotation matrix, not a vector
    normal = r_plane[:, 2]
    d = - (normal[0] * p[0] + normal[1] * p[1] + normal[2] * p[2])
    return create_extrinsic(r_plane, p), d, normal


def focal_ray_camworld(u, v, cam_mtx=CAM_MTX, cam_dist=CAM_DIST):
    """
    Return the 3D Ray vector going from the Camera center to the image point of coordinates (u,v).
    The ray has a z depth of the focal length,
    a x,y length of the number of pixels from its optical center multiplied by the length of a pixel
    """
    u, v = cv2.undistortPoints(np.array([u, v]), cam_mtx, cam_dist, P=cam_mtx)[0][0]
    return [(u - cam_mtx[0, 2]) * PIXSIZE, (v - cam_mtx[1, 2]) * PIXSIZE, FOCAL]


def get_camworld_point(point, plan_params, cam_mtx=CAM_MTX, cam_dist=CAM_DIST):
    """
    Get the coordinates of a 3d point on the camera World knowing the plan parameters the point
    reside on
    :param point: 2d pixel coordinate of the point
    :param plan_params: The plan parameters (Kext, d, normal)
    :param cam_mtx: The 3x3 camera intrinsic matrix
    :param cam_dist: The 5x1 camera distortion coef matrix
    :return: The point in 3d space on camera referential
    """
    u, v = point
    ray = focal_ray_camworld(u, v, cam_mtx, cam_dist)
    kext, d, normal = plan_params
    s = - d / (normal[0] * ray[0] + normal[1] * ray[1] + normal[2] * ray[2])
    return s * ray


def apply_extrinsic(xyz, kext):
    """
    Apply the extrinsic transformation for referential changing in 3d space
    :param xyz: point in old referential
    :param kext: Extrinsic parameters of the new referential
    :return: Point in new referential
    """
    k = kext.copy()
    coord = xyz.copy()
    las_xyz_camworld = np.array([coord[0], coord[1], coord[2], 1.])
    return np.dot(np.linalg.inv(k), las_xyz_camworld)[0:3]


def get_laser_plan_3d_pts(laser_cam_img_pts, chess_cam_img_pts, chess_plan_3d_pts, laser_plan_3d_pts, detected,
                          verbose):
    """
    Computes the laser dots coordinates on the chessboard plan referential for the Galvos-camera stereo_calibration.
    We use the chessboard image coordinates and the chessboard plan coordinates to extract the extrinsic parameters
    from the camera to the plan, then find the 3D coordinates in Camera referential and change the referential.

    If verbose is 3, compute the error for the Z projection of the chessboard plan points.

    :return: An array with laser dots coordinates on the chessboard plan referential and the plan parameters associated
    to each calibrations.
    """
    plan_params = []
    for k in range(len(chess_plan_3d_pts)):
        kext, d, normal = get_plan_param(chess_cam_img_pts[k], chess_plan_3d_pts[k])
        plan_params.append((kext, d, normal))

        if verbose > 2:
            error = []
            for i in range(len(chess_cam_img_pts[k])):
                xyz = get_camworld_point(chess_cam_img_pts[k][i][0], plan_params[k])
                xyz = apply_extrinsic(xyz, kext)

                err = abs(chess_plan_3d_pts[k][i] - xyz)
                error.append(err)
            print("Image ", detected[k])
            print("Mean projection Error in X: ", np.mean(error, axis=0)[0], "mm")
            print("Mean projection Error in Y: ", np.mean(error, axis=0)[1], "mm")
            print("Mean projection Error in Z: ", np.mean(error, axis=0)[2], "mm")

        error = []
        for i in range(len(laser_cam_img_pts[k])):
            laser_plan_3d_pts[k][i] = get_camworld_point(laser_cam_img_pts[k][i][0], plan_params[k])
            laser_plan_3d_pts[k][i] = apply_extrinsic(laser_plan_3d_pts[k][i], kext)
            # Our Z value should be 0 for the stereo_calibration. Our residue is the projection error
            error.append(laser_plan_3d_pts[k][i][2])
            laser_plan_3d_pts[k][i][2] = 0
        if verbose > 1:
            print("Z error for laser plan points projection: ", np.mean(error, axis=0), "mm")
            print()

    return laser_plan_3d_pts, plan_params


def extract_grid_galvworld_pts(img_pts, k_cam_galv, grid_plan3d_pts, cam_mtx=CAM_MTX, cam_dist=CAM_DIST, verbose=0):
    """
    Transform camera image coordinates into Galvos 3D coordinates
    :param img_pts: An image coordinates array
    :param grid_plan3d_pts: Coordinates of the dots on the plan
    :param k_cam_galv: 4x4 Galvos-Camera extrinsic matrix
    :param verbose: if 1 or more, show points
    :param cam_mtx: The 3x3 camera intrinsic matrix
    :param cam_dist: The 5x1 camera distortion coef matrix
    :return: An array of 3D coordinates in the Galvos referential
    """
    # Galvos referential shift from camera-galvos computed origin to Mechanical used Origin
    k_real_galv = create_extrinsic(GALVOS_ORIGIN_ROTATION, np.array([0, 0, 0], dtype=np.float64))

    plan_params = get_plan_param(img_pts, grid_plan3d_pts, cam_mtx, cam_dist)

    # Get their 3D camera coordinates
    grid_projworld_pts = np.zeros((len(img_pts), 1, 3), dtype="float32")
    for k in range(len(img_pts)):
        grid_projworld_pts[k] = get_camworld_point(img_pts[k][0], plan_params, cam_mtx, cam_dist)
        grid_projworld_pts[k] = apply_extrinsic(grid_projworld_pts[k][0], k_cam_galv)
        grid_projworld_pts[k] = apply_extrinsic(grid_projworld_pts[k][0], k_real_galv)
    # Normal plan change
    normal_projworld = plan_params[2]
    normal_projworld = np.dot(k_real_galv[0:3, 0:3], np.dot(k_cam_galv[0:3, 0:3], normal_projworld))

    if verbose > 1:
        print(grid_projworld_pts)
        print("Mean error in X:", np.mean(grid_projworld_pts, axis=0)[0][0])
        print("Mean error in Y:", np.mean(grid_projworld_pts, axis=0)[0][1])
        print("Mean error in Z:", np.mean(grid_projworld_pts, axis=0)[0][2] - GRID_DISTANCE)
        print()

    return grid_projworld_pts, normal_projworld


def generate_plan_coord(pattern_shape):
    # Our points are from left to right, top to bottom
    grid_plan3d_pts = np.zeros((pattern_shape[1] * pattern_shape[0], 3), dtype='float32')
    grid_plan3d_pts[:, :2] = np.mgrid[0:pattern_shape[0], 0:pattern_shape[1]].T.reshape(-1, 2)
    grid_plan3d_pts = grid_plan3d_pts * LASER_DISTANCE + [-(pattern_shape[0] - 1) * LASER_DISTANCE / 2,
                                                          -(pattern_shape[1] - 1) * LASER_DISTANCE / 2,
                                                          0]
    return grid_plan3d_pts
