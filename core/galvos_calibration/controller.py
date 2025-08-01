import os
import sys
import time
import mouse
import pyautogui
from enum import Enum

from core.utils.bcolors import bcolors
from global_config import FILES_DIR
from core.exceptions import FileSystemException


class ButtonType(Enum):
    Plus_X = 1
    Minus_X = 2
    Minus_Y = 3
    Plus_Y = 4
    Accept = 5
    Nudge = 6


class Controller:
    is_initialized: bool = False
    _nb_ignored: int = 1

    def __init__(self):
        if sys.platform == "linux":
            os.system("sudo xhost +")
        self.userClickPos = {}
        for bt_type in ButtonType:
            self.userClickPos[bt_type] = (-1, -1)

    def _on_click_lambda_nothing(self):
        pass

    def _on_click_lambda(self):
        """
        On click listener, add the click position if there are any empty values on [userClickPos]
        """
        x, y = mouse.get_position()
        if self._nb_ignored > 0:
            self._nb_ignored -= 1
            return
        for button_type in self.userClickPos:
            if self.userClickPos[button_type] == ():
                print(f"Click saved at {x}, {y}\n")
                self.userClickPos[button_type] = (x, y)
                break

    def setup(self, do_click=True, ask_load=True, save=True):
        """
        Do the setup until the user confirms he is satisfied
        """
        if ask_load:
            print(f"{bcolors.WARNING}Do you want to load last controller position? (y/n) \n"
                  "(only if the software window haven't moved from last time)")
            x = input()
            if x == 'y':
                self.load_controller_coord()
                self.is_initialized = True
                return
        try:
            self.setup_controller()
            self.confirm_setup(do_click)
            if not self.is_initialized:
                self.setup(do_click, False)
            else:
                if save:
                    self.save_controller_coord()
        except ImportError:
            print(f'{bcolors.FAIL}You must be root to use this library on linux.\n')

    def setup_controller(self):
        """
        Save the user left click to [userClickPos] in the [ButtonType] enum order and save them
        """
        self.userClickPos = {}
        for bt_type in ButtonType:
            self.userClickPos[bt_type] = ()

        self._nb_ignored = 1

        print("Welcome to the controller setup. Open the galvos calibration software\n"
              "to the point where you have the 5 buttons visible \n"
              "(Plus_X, Minus_X, Minus_Y, Plus_Y, Accept and Nudge) along with this command line, \n"
              "then write 'y' to continue. \n")
        x = input()
        while x != 'y':
            print("Type 'y' then press enter when you are ready")
            x = input()

        print(f"THE FIRST {self._nb_ignored} CLICK{'S' if self._nb_ignored > 1 else ''} WILL BE IGNORED.\n")
        mouse.on_click(self._on_click_lambda)
        for s in ButtonType:
            print(f"Left Click on the {s.name} Button")
            while self.userClickPos[s] == ():
                pass
        print()
        mouse.on_click(self._on_click_lambda_nothing)
        print("controller Setup done\n")

    def confirm_setup(self, do_click=True):
        """
        Test the stored position of mouse click, execute the routine on each point
        and ask for user satisfaction
        :return: True if the user press "y", False otherwise
        """
        print("Testing the setup. Ensure the controller clicks at the right positions\n")

        for buttonType in self.userClickPos:
            print(f"Moving to {buttonType} Button at {self.userClickPos[buttonType][0]}, "
                  f"{self.userClickPos[buttonType][1]}")
            mouse.move(self.userClickPos[buttonType][0], self.userClickPos[buttonType][1], absolute=True, duration=0.3)
            if do_click:
                print(f"Clicking... ")
                mouse.click()
            time.sleep(1)
            print()
        print(f"{bcolors.WARNING}Are you satisfied with the saved position? (y/n)")
        x = input()
        self.is_initialized = x == "y"

    def click_button(self, button: ButtonType, instant=True):
        """
        Click on the button position the controller stored during setup.
        :param instant: Make the click instantaneous if True, add a small delay if False
        :param button: The button to click
        """
        if self.userClickPos[button] == ():
            print("Setup the controller before trying to click")
            print(f"Go {button.name}")
            return
        t = 0 if instant else 0.1
        mouse.move(self.userClickPos[button][0], self.userClickPos[button][1], absolute=True, duration=t)
        mouse.click()

    def load_controller_coord(self):
        file_path = os.path.join(FILES_DIR, "controller.txt")
        if not os.path.isfile(file_path):
            raise FileSystemException("Previous ground truth file not found")

        with open(file_path, 'r') as controller_file:
            lines = controller_file.readlines()
            for line in lines:
                button, xy = line.split(':')
                x, y = xy.split(',')
                self.userClickPos[ButtonType(int(button))] = (int(x), int(y))

    def save_controller_coord(self):
        with open(os.path.join(FILES_DIR, "controller.txt"), 'w') as controller_file:
            for buttonType in self.userClickPos:
                controller_file.writelines(f"{buttonType.value}:"
                                           f"{self.userClickPos[buttonType][0]},"
                                           f"{self.userClickPos[buttonType][1]}\n")

    @staticmethod
    def keyboard_write(text: str):
        """ Write something with the keyboard """
        try:
            pyautogui.typewrite(text)
        except pyautogui.FailSafeException:
            print(f"{bcolors.FAIL} Controller clicked outside of screen for mysterious reason")

    @staticmethod
    def keyboard_clear_field(n: int = 5):
        """ Press delete button several times"""
        try:
            if n < 0:
                n = 1
            pyautogui.press('delete', presses=n)
            pyautogui.press('backspace', presses=n)
        except pyautogui.FailSafeException:
            print(f"{bcolors.FAIL} Controller clicked outside of screen for mysterious reason")

    @staticmethod
    def click(x, y):
        try:
            mouse.move(x, y)
            mouse.click()
        except pyautogui.FailSafeException:
            print(f"{bcolors.FAIL} Controller clicked outside of screen for mysterious reason")


if __name__ == '__main__':
    cont = Controller()
    cont.setup(ask_load=True)
    cont.click_button(button=ButtonType.Nudge)
    time.sleep(2)
    cont.click_button(button=ButtonType.Nudge)
    cont.keyboard_write("Walla")
