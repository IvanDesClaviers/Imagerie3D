import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(ROOT_DIR, "files")
ENV_PATH = os.path.join(ROOT_DIR, '.env')
GALVOS_SETUP_DIR = os.path.join(FILES_DIR, "galvos_calib_setup")
GALVOS_MASK_FILE = os.path.join(GALVOS_SETUP_DIR, "ground_truth_temp.txt")
IMG_DIR = os.path.join(FILES_DIR, "images")
GALVOS_CAMERA_CALIB_DIR = os.path.join(GALVOS_SETUP_DIR, "camera_calib")

