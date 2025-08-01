import numpy as np
from cv2 import cv2

# 200 Hz
# 10 fps
# max expo


###############################
# PATH VARIABLES
###############################
GRID_LASER_IMG_PATH = "/home/ivan/Work/Images/image_test.bmp"
CAMERA_CALIB_PATH = "/home/ivan/Work/Images/chess"  # "calibration_images/chessboard"
CAMERA_GALV_CALIB_PATH = "/home/ivan/Work/Images/proj"  # calibration_images/galvos"

###############################
# CALIBRATION SETUP PARAMETERS
###############################
CHESSBOARD_SHAPE = (7, 5)
LASER_PATTERN_SHAPE = (7, 7)  # You must project a rectangle shaped pattern of dots

SQUARE_SIZE = 40  # Square length in the chessboard [mm]
LASER_DISTANCE = 100  # The distance between each laser dot on the grid [mm]
GRID_DISTANCE = 2015.5  # Distance from the Galvos mechanical center to the center of the grid [mm]
GRID_SIZE = (2000, 2000)  # Size of the Grid [mm]

# Since we cannot project the pattern on the full available area and see the pattern on the chessboard plane at
# the same time, we have to reduce the projection area, and thus create an offset on the projection.
PROJECTION_OFFSET = (-300, -300)  # Offset of the projected pattern top left corner in the camera referential [mm]

###############################
# CAMERA PARAMETERS
###############################
RESOLUTION = tuple((2076, 3088))  # Image resolution [pix], modify only if you change the camera sensor
PIXSIZE = 0.0024  # Size of a pixel [mm], modify only if you change the camera sensor

# Camera Calibration Results
CAM_MTX = np.array([[2.09476635e+03, 0.00000000e+00, 1.55589161e+03],
                    [0.00000000e+00, 2.09539431e+03, 1.03175612e+03],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

CAM_DIST = np.array([[-3.45944413e-01],
                     [1.69194959e-01],
                     [-9.54054758e-05],
                     [-6.37788842e-04],
                     [-5.44604897e-02]])

FOCAL = ((CAM_MTX[0, 0] + CAM_MTX[1, 1]) * PIXSIZE / 2)  # [mm]

###############################
# GALVOS CALIBRATION
###############################
# Galvos-Camera Calibration result, to avoid calibrating everytime.
# It is the default value for the last configuration, you HAVE to edit it if you change
# the position of the camera and/ or the galvos
KEXT_GUESS = np.array([[9.9984676e-01, -1.7455125e-02, 1.3321623e-03, -3.2414543e+01],
                       [1.7374881e-02, 9.9877930e-01, 4.6239767e-02, -3.1571030e+01],
                       [-2.1376570e-03, -4.6209536e-02, 9.9892944e-01, -2.2607885e+01],
                       [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])

# Volume where we expect to find the Galvos theoretical center.
# It is the default value for the last configuration, you HAVE to edit it if you change
# the position of the camera and/ or the galvos
# Values are to be given on Camera referential
GALVOS_ZONE = np.array([[-25, -45], [-40, -20], [-40, -10]])

# Rotation Matrix to change from Camera referential to Galvos referential
GALVOS_ORIGIN_ROTATION = np.array([[0, 1, 0],
                                   [-1, 0, 0],
                                   [0, 0, 1]])

###############################
# MISC CONSTANTS
###############################

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.001)  # DO NOT CHANGE
