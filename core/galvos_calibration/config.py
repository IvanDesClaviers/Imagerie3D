from core.cameras import CamType

MAX_ITER = 1000  # [iteration]
GRID_SIZE = 2000  # [mm]
NB_MAX_GUESS = 40  # [iteration]

# CAMERA DATA
RESOLUTION = tuple((3684, 4912))  # Image resolution [pix]
PIXSIZE = 0.0024  # Size of a pixel [mm]
SERIAL_NUMBER_ROLE = {b'4103846968': CamType.TOP_RIGHT,
                      b'4103846966': CamType.TOP_LEFT,
                      b'4103846969': CamType.BOT_RIGHT,
                      b'4103762733': CamType.BOT_LEFT}

# GALVOS CALIBRATION DATA
MAX_CALIB_NUM = 7
NB_POINT_PER_CALIB = {1: 3 ** 2, 2: 5 ** 2, 3: 9 ** 2, 4: 11 ** 2, 5: 21 ** 2, 6: 41 ** 2, 7: 81 ** 2}
MAX_NB_PT_LINE = 81
GALVOS_DISTANCE_FROM_ORIGIN = 2017.1
GALVOS_MAX_MECHANICAL_ANGLE = 15
MAX_GALVOS_NUDGE = 250
GALVOS_STEP_SPEED = 0.1
MEAN_CALIBRATION_TIME_PER_POINT = 10  # [sec]
