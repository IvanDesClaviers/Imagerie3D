import argparse

from core import *

parser = argparse.ArgumentParser()

####################################################################################################
# Plan Calibration arguments
####################################################################################################
parser.add_argument("-calibrate-plan",
                    help="Launch camera stereo_calibration",
                    action="store_true")

parser.add_argument("--cam-path",
                    help="Specify camera stereo_calibration path",
                    type=str,
                    default=CAMERA_CALIB_PATH)

parser.add_argument("--galv-path",
                    help="Specify camera-galvos stereo_calibration path",
                    type=str,
                    default=CAMERA_GALV_CALIB_PATH)

parser.add_argument("--angles",
                    help="Path to the grid image with the projected pattern on it in order to compute the grid angles",
                    type=str)

####################################################################################################
# Galvos Correction arguments
####################################################################################################

parser.add_argument("-galvos-calib",
                    help="Launch the laser galvos_calibration module",
                    action="store_true")

####################################################################################################
# Camera Calibration to Galvos and Rangefinder arguments
####################################################################################################

parser.add_argument("-calibrate-ids-camera-to-laser-and-galvos",
                    help="Launch the laser galvos_calibration module",
                    action="store_true")
####################################################################################################
# Global arguments
####################################################################################################


parser.add_argument("-v", "--verbosity",
                    action="count",
                    default=4,
                    help="increase output verbosity")

parser.add_argument("--cam",
                    help="Chose the camera type to use. Possible values are:"
                         " '1' for ids camera,"
                         " '2' for 1st camera on the camera list (Opencv)",
                    type=int,
                    default=1)

args = parser.parse_args()

if __name__ == '__main__':
    from core import *

    if args.calibrate_plan and args.angles:
        cam_mtx, cam_dist, k_cam_galv, galv_mtx, galv_dist_coef = calibrate(args.cam_path, args.galv_path,
                                                                            args.verbosity)
        find_angle_grid(args.angles, k_cam_galv=k_cam_galv, cam_mtx=cam_mtx, cam_dist=cam_dist, verbose=args.verbosity)

    if not args.calibrate_plan and args.angles:
        find_angle_grid(args.angles, verbose=args.verbosity)

    if args.galvos_calib:
        galvosCalibrator = GalvosCalibrator(args.verbosity, args.cam)
        galvosCalibrator.calibrate()

    if args.calibrate_ids_camera_to_laser_and_galvos:
        cam = IdsCamera()
        cam.show_camera_stream()
