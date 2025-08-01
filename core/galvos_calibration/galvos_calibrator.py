from core.galvos_calibration import *


class GalvosCalibrator:
    controller: Controller  # our mouse / keyboard controller
    cameras: MultipleCameras  # The cameras
    _cam_type: int  # Camera type used
    _nb_grid_calib: int = 0  # The number of grid points to calibrate
    _pixel_criteria: int = 0  # The accuracy criteria in pixel
    _curr_nudge: int = -1  # The current speed of the software
    _nb_guess: int = 0  # The number of time we guessed a direction during auto-calibration
    _terminated: bool = False  # Indicates if the task is interrupted by the user
    _task_hold: bool = False  # Indicates if the task is paused
    _start_time: float = 0.0  # Keep track of the time it took to perform the calibration
    _quick: bool = False  # Set to true to automatically load everything when available and skip confirmations
    ground_truth: np.array = np.zeros((MAX_NB_PT_LINE, MAX_NB_PT_LINE), dtype=tuple)  # The data for calibration
    _already_configured = False  # Track of global internal state
    _dummy: bool = False  # If true, doesn't enable camera detection

    def __init__(self, save=True, quick=False, dummy=False):
        self._cam_type = 2 if dummy else 1
        self._save = save
        self.controller = Controller()
        self._quick = quick
        self._dummy = dummy
        for i in range(MAX_NB_PT_LINE):
            for j in range(MAX_NB_PT_LINE):
                self.ground_truth[i, j] = (CamType.BOT_LEFT, (-1, -1))

    def calibrate(self):
        """
        This is the main function of the calibration. It launches the calibration procedure,
        and allow the user to calibrate several tables one after the other
        """
        print(f"{bcolors.OKGREEN}Hi, this is the galvos calibration corrector. \n"
              f"{bcolors.OKGREEN}This script will perform the galvos calibration automatically for you. \n"
              f"{bcolors.OKGREEN}Please Follow the next step to ensure the galvos calibration works successfully.\n")
        self.calibrate_table()

        print(f"{bcolors.WARNING}Would you like to calibrate another table? (y/n)")
        x = input()
        redo_calib = True
        while redo_calib:
            while x != 'y' and x != 'n':
                print(f"{bcolors.WARNING}Would you like to calibrate another table? (y/n)")
                x = input()
            if x == 'y':
                self.calibrate_table()
                x = ''
            else:
                redo_calib = False
        print(f"{bcolors.OKGREEN}Galvos Calibration is over, see you next time ;) ")
        self.cameras.close_all()

    def calibrate_table(self):
        """
        This function perform the whole calibration of 1 table
        It calls each steps one by one, initialize everything and start the galvos calibration once ready
        """
        self._chose_nb_galvos_table()
        self.configure_calibrator()
        nb_pts_per_line = int(sqrt(NB_POINT_PER_CALIB[self._nb_grid_calib]))
        sub_gt = GroundTruthManager.extract_sub_ground_truth(self.ground_truth, nb_pts_per_line)
        if not self._dummy:
            time_before_start = 5
            print(f"\n{bcolors.OKGREEN}Thank you for following all the steps. Let's make sure everything is ready!\n\n"
                  ""
                  f"{bcolors.OKBLUE}Use the software to put the laser at its first position, and check once again:\n"
                  f"{bcolors.OKBLUE}- The scale is x1,\n"
                  f"{bcolors.OKBLUE}- The grid size is {GRID_SIZE} mm,\n"
                  f"{bcolors.OKBLUE}- The Max Mechanical angle is set to {GALVOS_MAX_MECHANICAL_ANGLE} degree,\n"
                  f"{bcolors.OKBLUE}- The focal point distance is set at {GALVOS_DISTANCE_FROM_ORIGIN} mm,\n"
                  f"{bcolors.OKBLUE}- The computer is powered for the whole process,\n"
                  f"{bcolors.OKBLUE}- You have started calibration for {nb_pts_per_line}x{nb_pts_per_line} points\n"
                  f"{bcolors.OKBLUE}- No one will walk in front of it\n"
                  f"{bcolors.OKBLUE}- The lights are turned off\n"
                  f"{bcolors.OKBLUE}- The laser voltage is set to 5.58 V\n"
                  f"{bcolors.OKBLUE}- The laser intensity is set to 0.065 A\n"
                  f"{bcolors.OKBLUE}- The volume of the computer is on\n"
                  f"{bcolors.OKBLUE}- The 'sleep mode' is disabled on the os you are on\n"
                  ""
                  f"{bcolors.OKBLUE}You need to be save everything manually, "
                  "so come check how the software is doing occasionally,"
                  " but don't touch the mouse for any unexpected behavior.\n"
                  ""
                  f"{bcolors.OKBLUE}The expected time for this calibration is "
                  f"{MEAN_CALIBRATION_TIME_PER_POINT * NB_POINT_PER_CALIB[self._nb_grid_calib] / 60} min\n"
                  ""
                  f"{bcolors.OKBLUE}When ready to start the galvos Calibration,"
                  " type 'ready' in the command line and press enter. \n"
                  f"You then have {time_before_start} seconds to open the calibration software "
                  f"by NOT MOVING the window.\n")

            x = input()
            while x != 'ready':
                print(f"{bcolors.OKBLUE}When ready to start the galvos Calibration, type 'ready' "
                      f"in the command line and press enter. \n"
                      f"You then have {time_before_start} seconds to open the calibration"
                      f" software by NOT MOVING the window.\n")
                x = input()
            for k in range(time_before_start):
                print(f"{bcolors.WARNING}Starting in {time_before_start - k}...")
                time.sleep(1)

            print(f"{bcolors.OKGREEN}Launching the Galvos Calibration! Wish me luck!\n\n")
            print(f"{bcolors.BOLD}To pause the controller from clicking, press 'SPACE'\n"
                  f"{bcolors.BOLD}To skip a point, pause the controller and then press 'ALT'\n"
                  f"{bcolors.BOLD}To quit, press 'ESC'\n")
        self._start_time = time.time()
        self._perform_calibration(sub_gt)
        total_time = time.time() - self._start_time
        print(f"{bcolors.OKGREEN}Total elapsed time: {total_time / 60} minutes\n"
              f"{bcolors.OKGREEN}Average time per point: "
              f"{total_time / (60 * NB_POINT_PER_CALIB[self._nb_grid_calib])} minutes!\n\n")

    def configure_calibrator(self):
        """
        Load all necessary setup once, such as the camera setup, the mouse controller
        and the whole detection of the grid
        :return:
        """
        if self._already_configured:
            return
        print("Configuring the cameras...\n")
        self.cameras = MultipleCameras(cam_mode=self._cam_type)
        if not self._quick:
            print(f"{bcolors.OKBLUE}We will now show all cameras one by one.\n"
                  "When you are satisfied by the setup, click on the image\n"
                  "and press any key.\n")
            self.cameras.test_cameras_one_by_one()

        print("Loading mouse controller...\n")
        if self._quick:
            self.controller.load_controller_coord()
        else:
            self.controller.setup(False, save=self._save)
        if not self._dummy:
            self._create_ground_truth()
        self._already_configured = True

    def _chose_nb_galvos_table(self):
        """
        Select the configuration for the calibration, aka the number of point to calibrate
        """
        if self._quick:
            self._nb_grid_calib = self._nb_grid_calib + 1
            return
        print("First, enter the number of the configuration want to perform. "
              f"Number must be between 1 and {MAX_CALIB_NUM}\n")
        for k in range(1, len(NB_POINT_PER_CALIB) + 1):
            n = int(math.sqrt(NB_POINT_PER_CALIB[k]))
            print(f"{bcolors.BOLD}{k} -> {n}x{n} ({NB_POINT_PER_CALIB[k]} points)")

        x = input()
        proceed = False
        while not proceed:
            try:
                self._nb_grid_calib = int(x)
                if 0 < self._nb_grid_calib <= MAX_CALIB_NUM:
                    proceed = True
                else:
                    print(f"The input number must be between 0 and {MAX_CALIB_NUM}")
                    x = input()
            except ValueError:
                print(f"The input number must be between 0 and {MAX_CALIB_NUM}")
                x = input()

    def _create_ground_truth(self):
        """ Launch the creation of the ground_truth """
        if self._quick:
            load = self._quick
        else:
            load = None
            while load is None:
                print(f"{bcolors.WARNING}Would you like to load the previous detected quadrants? (y/n)")
                x = input()
                if x == 'y':
                    load = True
                elif x == 'n':
                    load = False
        cam_with_center = MultipleCameras.chose_new_center(False)
        self.ground_truth = GroundTruthManager().create_ground_truth(self.cameras.cam_dict, cam_with_center,
                                                                     save=True, load=load)

    def _perform_calibration(self, ground_truth):
        """
        This method will go through the ground truth in the correct order
        and automatically calibrate all points with the controller.
        Everything needs to have been set up correctly before starting this function.
        :param ground_truth:  The ground_truth is a matrix of size NxN, where N is the number of points
        to calibrate on the grid. Its value is a tuple (CamType, (x_pos, y_pos)) describing, for each point,
        which camera is in charge of calibrating it and what is the desired position in the image.
        """
        # This puts the coordinates to correct in the right order, meaning right to left and top to bottom
        nb_pt_on_line = ground_truth.shape[0]
        ground_truth_list = [ground_truth[j, nb_pt_on_line - i]
                             for j in range(nb_pt_on_line) for i in range(1, nb_pt_on_line + 1)]
        cameras_quadrant_masks = self.get_cameras_masks()

        for pt_id in range(len(ground_truth_list)):
            if self._terminated:
                print(f"{bcolors.WARNING}Terminated by user\n")
                self.cameras.close_all()
                return
            # Load setup
            camera_type, p_intersect = ground_truth_list[pt_id]
            default_direction = self._get_default_direction(nb_pt_on_line, pt_id, camera_type)
            print(f"Default direction is {default_direction}")

            camera = self.cameras.cam_dict[camera_type]
            quad_mask = cameras_quadrant_masks[camera_type]

            print(f"{bcolors.HEADER}Correcting using the {camera_type} Camera...")
            self._correct_point_automatically(camera, p_intersect, default_direction, quad_mask)
            print(f"{bcolors.OKGREEN}Point number {pt_id + 1} corrected!")
            print(f"{bcolors.OKCYAN}Elapsed time: {(time.time() - self._start_time) / 60} minutes\n")

    def _correct_point_automatically(self, camera: Camera, p_intersect: tuple, default_direction: tuple,
                                     quad_mask: np.array):
        """
        This method will compare the distance between our p_intersect and the laser dot center.
        If the controller is initialized, it will automatically click, otherwise it will only print the
        information.
        The intersection point, the size of a pixel and ou criteria must be initialized.
        :param camera: The opened camera we use for the video stream
        :param p_intersect: The desired coordinates extracted from the ground truth
        :param default_direction: A vector for the direction to click. Positive is Right/Down.
        """
        if self._dummy:
            return
        from pynput.keyboard import Listener
        # Reset
        camera.start_video()
        task_finished = False
        self._task_hold = False
        keyboard_press_listener = Listener(on_press=self._on_press)
        keyboard_press_listener.start()
        k = 0
        time.sleep(1)
        self._nb_guess = 0
        print(f"{bcolors.BOLD}Target point coordinate: {p_intersect}\n")
        while camera.is_open() and not task_finished and not self._terminated and k < MAX_ITER:
            if not self._task_hold:
                frame = camera.get_frame()
                task_finished = self._click_on_software(frame, p_intersect, default_direction, quad_mask)
                print()
                k += 1
        if k == MAX_ITER:
            print(f"{bcolors.FAIL}CALIBRATION TOOK TOO LONG, going next")
            self.controller.click_button(ButtonType.Accept)

        keyboard_press_listener.stop()
        camera.stop_video()

    def _click_on_software(self, frame, p_intersect: tuple, default_direction: tuple, quad_mask: np.array):
        """
        The threaded callback computing the current distance between our intersection and
        the detected laser center. If we are within the criteria range, set task_finished to True
        :param frame: The current frame
        :param p_intersect: The target point we want to reach
        :param default_direction: A vector for the direction to click. Positive is Right/Down.
        :return: True if we are in range, False if not
        """
        try:
            p_laser = find_laser(frame, quad_mask)
            dist = (round(p_intersect[0] - p_laser[0]), round(p_intersect[1] - p_laser[1]))
            print(f"Current distance: {dist}")
            return self._auto_click(self.get_nudge(dist), dist)
        except ComputerVisionException:
            # It it is our first fail attempt we go to the default direction
            max_nb_guess = NB_MAX_GUESS // self._nb_grid_calib
            if self._nb_guess < max_nb_guess:
                self._guess_click(default_direction)
                self._nb_guess += 1
            else:
                print(f"{bcolors.FAIL} Couldn't detect laser after {max_nb_guess} guess, the laser is most likely "
                      f"not visible because the intensity is too low, please augment it.")
                if sys.platform == "linux":
                    os.system('play -nq -t alsa synth {} sine {}'.format(1, 440))
                else:
                    import winsound
                    winsound.Beep(440, 1000)
            return False

    def _auto_click(self, nudge: list, dist: tuple):
        """
        Automatically click on the setup buttons in order to correct the points on the grid
        if the controller is initialized.
        :param dist: The tuple (horizontal distance, vertical distance) in pixel
        :param nudge: A list 1x2 representing the nudge to set horizontally and vertically
        :return: True if we are in range, False if not
        """
        for k in range(2):
            if self._task_hold or self._terminated:
                return False
            if abs(dist[0]) <= self._pixel_criteria and abs(dist[1]) <= self._pixel_criteria:
                print(f"Laser is in range, clicking on button Accept\n")
                self.controller.click_button(ButtonType.Accept)
                return True

            if nudge[k] != 0:
                if nudge[k] != self._curr_nudge:
                    self._set_software_nudge(nudge[k])
                if dist[k] > 0:
                    button = ButtonType.Minus_X if k == 0 else ButtonType.Plus_Y
                else:
                    button = ButtonType.Plus_X if k == 0 else ButtonType.Minus_Y
                print(f"Clicking on button {button.name}")
                self.controller.click_button(button)
        print()
        return False

    def get_nudge(self, dist: tuple):
        """
        Adjust the speed so we need only up to 5 clicks to approach greatly the destination
        :param dist: The tuple (horizontal distance, vertical distance) in pixel
        :return: A list 1x2 representing the nudge to set horizontally and vertically
        """
        nudge = [0, 0]
        lim = 15  # 15 is arbitrary and has been found empirically
        for k in range(2):
            if self._pixel_criteria < abs(dist[k]) < lim:
                nudge[k] = GALVOS_STEP_SPEED
            else:
                nudge[k] = abs(dist[k]) // lim if abs(dist[k]) // lim < MAX_GALVOS_NUDGE else MAX_GALVOS_NUDGE
        return nudge

    def _on_press(self, key):
        """ The listener to pause on SPACE press and quit on ESC press """
        from pynput.keyboard import Key
        if key == Key.space:
            self._task_hold = not self._task_hold
            if self._task_hold:
                print(f"{bcolors.WARNING}Press space again to stop the pause")
            else:
                print(f"{bcolors.WARNING}Resuming...")
        elif key == Key.esc:
            print(f"{bcolors.FAIL}We stop EVERYTHING")
            self._terminated = True

    def _guess_click(self, default_direction: tuple):
        """
        Click on the software without knowing the distance between the laser and its target point.
        :param default_direction: A vector for the direction to click. Positive is Right/Down.
        """
        self._set_software_nudge(20)  # Arbitrary value

        for k in range(2):
            if self._task_hold or self._terminated:
                return
            if default_direction[k] != 0:
                if default_direction[k] > 0:
                    button = ButtonType.Minus_X if k == 0 else ButtonType.Plus_Y
                else:
                    button = ButtonType.Plus_X if k == 0 else ButtonType.Minus_Y
                self.controller.click_button(button)
                print(f"{bcolors.WARNING}Laser not visible, guessing click: {button.name}")

    def _set_software_nudge(self, nudge_val: int):
        """
        Edit the nudge value in the software
        :param nudge_val: The speed we want to set
        """
        print(f"{bcolors.WARNING}Going to speed = {nudge_val}")
        self.controller.click_button(ButtonType.Nudge)
        self.controller.keyboard_clear_field(3)
        self.controller.keyboard_write(str(nudge_val))
        self._curr_nudge = nudge_val

    def get_cameras_masks(self):
        """
        Load the camera masks for each camera
        :return: A dict, for each camera type its associated list of coordinates
        """
        cam_quad: dict = {CamType.TOP_RIGHT: None,  # The quadrant mask for each camera
                          CamType.TOP_LEFT: None,
                          CamType.BOT_RIGHT: None,
                          CamType.BOT_LEFT: None}

        for c in CamType:
            camera = self.cameras.cam_dict[c]
            camera.start_video()
            try:
                im = camera.get_frame()
                cam_quad[c] = np.zeros(im.shape[0:2], dtype=np.uint8)
                cv2.fillPoly(cam_quad[c], [np.array(GroundTruthManager.load_quadrant_mask(c))], 255)
            except (FileNotFoundError, AttributeError):
                print(f"{bcolors.WARNING}Galvos Calibrator: Could not load saved mask for camera {c}")
            finally:
                camera.stop_video()

        return cam_quad

    @staticmethod
    def _get_default_direction(n: int, pt_id: int, cam_type: CamType, nb_cam=4):
        """
        Find the default direction we should go if no laser is found
        Note: There is probably a direct mathematical way to achieve it
        :param n: The number of point in a line
        :param pt_id: The point id to evaluate from 0 to nxn -1
        :param cam_type: The current camera type
        :param nb_cam:
        :return: A tuple (default_direction_in_x, default_direction_in_y)
        """
        # The default direction is toward the center of a quadrant
        # The number of quadrant is the number of camera
        x_pos = pt_id % n
        y_pos = pt_id // n
        mid = (n - 1) / 2

        default_x = (-1) ** (1 + x_pos * nb_cam // (2 * n))
        default_y = (-1) ** (y_pos * nb_cam // (2 * n))
        # We apply the special rule on borders and for the central point
        if y_pos == mid == x_pos:  # Central point is special
            default_x = - 1 if cam_type == CamType.BOT_LEFT or cam_type == CamType.TOP_RIGHT else 1
            default_y = -1 if cam_type == CamType.TOP_RIGHT or cam_type == CamType.TOP_LEFT else 1
        elif y_pos == mid and x_pos != mid:  # Vertical middle axis
            return default_x, -1 if x_pos < mid else 1
        elif y_pos != mid and x_pos == mid:  # Horizontal middle axis
            return -1 if y_pos < mid else 1, default_y
        return default_x, default_y
