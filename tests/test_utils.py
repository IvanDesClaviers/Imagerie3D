import unittest

from global_config import *
from core.galvos_calibration.controller import Controller


class TestUtils(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestUtils, self).__init__(*args, **kwargs)

    def test_click_and_write(self):
        os.system("sudo ls -A")
        os.system("xhost +")
        c = Controller()
        c.click(100, 100)
        c.keyboard_write("Walla")
