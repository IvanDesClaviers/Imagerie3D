import gc
import time
import matplotlib

from core.exceptions import *
from core.galvos_calibration.config import *
from core.galvos_calibration.grid_processing import *


class GroundTruthManager:
    # Saved variables to have them accessible through the callbacks
    _frame = []  # The active frame we are using
    _backup_frame = []  # The saved frame with all intersections on it
    _quadrant_cnt = []  # The coordinate of the quadrant on the active frame
    _shower: matplotlib.image.AxesImage  # The shower

    def __init__(self):
        matplotlib.use('TkAgg')

    def create_ground_truth(self, cams: dict, cam_with_center=CamType.BOT_LEFT, save=True, load=False):
        """
        Create an interactive image so the user can set up the ground truth matrix.
        which represents the coordinates of the ground truth the camera is in charge of.
        :param cams: The dictionary <CamType, Camera> of all the camera and their role
        :param cam_with_center: The camera type which will take the center point
        :param save: For each camera, load the last quadrant delimitation
        :param load: For each camera, try to load previous config file
        :return: The ground truth matrix of size NxN. Its value is a tuple (CamType, (x_pos, y_pos)) describing,
        for each point, which camera is in charge  of calibrating it and what is the desired position in the image.
        """
        mid = (MAX_NB_PT_LINE - 1) // 2
        end = MAX_NB_PT_LINE - 1
        n_quad = (MAX_NB_PT_LINE - 1) // 2 + 1  # The number of point in a line of the quadrant

        cam_coord = {CamType.TOP_RIGHT: ((0, mid), (mid + 1, end)),
                     CamType.TOP_LEFT: ((0, mid - 1), (0, mid)),
                     CamType.BOT_RIGHT: ((mid + 1, end), (mid, end)),
                     CamType.BOT_LEFT: ((mid, end), (0, mid - 1))}

        ground_truth: np.array = np.zeros((MAX_NB_PT_LINE, MAX_NB_PT_LINE), dtype=tuple)

        print(f"\n{bcolors.OKBLUE}Welcome in the Ground truth generator!\n"
              f"{bcolors.OKBLUE}Each camera will show a quadrant of the calibration grid.\n"
              f"{bcolors.OKBLUE}To generate the ground truth, you need to delimit the quadrants,\n"
              f"{bcolors.OKBLUE}so we ask you to right click around the 41x41 points the camera has to detect\n"
              f"{bcolors.OKBLUE}and to exclude any other points."
              f"{bcolors.OKBLUE}This operation is critical to the accuracy of the galvos calibration!\n"
              f"\n {bcolors.OKBLUE}When the window is open, you can: \n"
              f"{bcolors.OKBLUE}- LEFT CLICK to navigate or zoom in the image,\n"
              f"{bcolors.OKBLUE}- RIGHT CLICK to select a point and create a line (It may not appear), \n"
              f"{bcolors.OKBLUE}- MIDDLE CLICK to delete the last line.\n")

        for cam_type in cam_coord.keys():
            # Initialization
            print(f"Generation of the ground_truth for the {cam_type} Camera ...")
            cam = cams[cam_type]
            cam.start_video()
            time.sleep(1)

            # Image processing
            original_frame = cam.get_frame()
            self._backup_frame, intersects = get_all_grid_intersections(original_frame)
            quad_pts, quad_im, quad_cnt = self._detect_quadrant_intersections(intersects, cam_type, n_quad, save, load)
            GroundTruthManager.edit_ground_truth(quad_pts, ground_truth, n_quad, cam_coord, cam_type, cam_with_center)

            # Clearing
            cam.stop_video()
            gc.collect()  # Avoid memory overflow
            cv2.destroyAllWindows()
            print(f"{bcolors.OKCYAN}Generation for {cam_type} Camera done!\n")

        return ground_truth

    def _detect_quadrant_intersections(self, intersections: list, cam_type: CamType, n: int, save: bool, load: bool):
        """
        Create the interactive window to select a quadrant.
        Once created, it takes all detected coordinates and sort them into a NxN matrix, filling the blanks.
        If an error happen, it asks to redo it.
        :param intersections: The list of all cross intersections on the image, result of image processing
        :param cam_type: The current camera type
        :param n: The number of point in a line
        :param save: True if the quadrant contours are saved when successful
        :param load: True if we try to load the previous coordinates instead of using the interactive window
        :return:
        """
        im_shape = (self._backup_frame.shape[0], self._backup_frame.shape[1], 3)
        while True:
            try:
                self._frame = self._backup_frame.copy()
                self._quadrant_cnt = self.load_quadrant_mask(cam_type) if load else self._delimit_quadrant()
                # Order points
                quadrant_img, quad_intersections, quadrant_mask_cnt = self._extract_quadrant_data(intersections)
                all_quadrant_points = get_calibration_intersections(quad_intersections, n, im_shape, True)

                if save:
                    self.save_quadrant_mask(cam_type, quadrant_mask_cnt)
                return all_quadrant_points, quadrant_img, quadrant_mask_cnt
            except (ComputerVisionException, IndexError, FileNotFoundError):
                load = False
                print(f"{bcolors.WARNING}Automatic detection failed, please try again")

    def _delimit_quadrant(self):
        """ Create the matplotlib image with the callback on click in order to delimit the quadrant """
        # Put the image in full screen with no white border
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        self._shower = plt.gca().imshow(self._frame, aspect='auto', cmap='gray', interpolation='nearest')

        cid = plt.gcf().canvas.mpl_connect('button_press_event', self._mouse_click)  # Setup the interactive window
        plt.show()
        plt.gcf().canvas.mpl_disconnect(cid)  # Disconnect the callbacks after the user has finished
        return self._quadrant_cnt

    def _mouse_click(self, event):
        """
        The callback of the mouse clicks on the plotted image.
            - Right click selects a point for the quadrant contour
            - Middle click pops the last point selected
        :param event: The variable holding the position on the image of the click coordinates (mostly)
        """
        if event.button == 2 and len(self._quadrant_cnt) > 0:  # Middle click: Pop the last point
            # Pop point
            self._quadrant_cnt.pop()
            self._frame = self._backup_frame.copy()
            # Redraw previous lines
            for i in range(1, len(self._quadrant_cnt)):
                cv2.line(self._frame, self._quadrant_cnt[i - 1], self._quadrant_cnt[i], 255, lineType=cv2.LINE_AA)
            # Refresh image
            self._shower.set_data(self._frame)
            plt.draw()
        elif event.button == 3:  # Right click: add point
            try:
                # Save point
                self._quadrant_cnt.append((int(event.xdata), int(event.ydata)))
                # Draw line
                if len(self._quadrant_cnt) > 1:
                    cv2.line(self._frame, self._quadrant_cnt[-2], self._quadrant_cnt[-1], 255, lineType=cv2.LINE_AA)
                # Refresh image
                self._shower.set_data(self._frame)
                plt.draw()
            except TypeError:
                print("Don't right click outside of the image dummy")

    def _extract_quadrant_data(self, intersections):
        """
        Apply _quadrant_cnt on _frame and extract all date from it
        :param intersections: The list of all cross intersections in our frame
        :return: The tuple (quadrant_img, quadrant_intersections, mask_cnt) with:
        - quadrant_img the image of the delimited quadrant on the frame
        - quadrant_intersections the list of intersections contained on the quadrant
        - mask_cnt the coordinated of the mask's edges
        """
        mask = np.zeros(self._frame.shape[0:2], dtype=np.uint8)
        mask_cnt = [np.array(self._quadrant_cnt)]
        self._quadrant_cnt = []
        cv2.fillPoly(mask, mask_cnt, 255)
        quadrant_img = cv2.bitwise_and(mask, self._backup_frame)

        quadrant_intersections = []
        for pt in intersections:
            if quadrant_img[pt[1], pt[0]] != 0:
                quadrant_intersections.append(pt)

        return quadrant_img, quadrant_intersections, mask_cnt

    @staticmethod
    def edit_ground_truth(coord_mat: np.array, ground_truth: np.array, nb_pt_line_quad: int, cam_coord: dict,
                          cam_type: CamType, cam_with_center: CamType):
        """
        :param coord_mat: The matrix of coordinates of the intersections
        :param ground_truth:  The ground truth matrix of size NxN, with N the number of point in a line.
        Its value is a tuple (CamType, (x_pos, y_pos)) describing, for each point,
        which camera is in charge  of calibrating it and what is the desired position in the image.
        :param nb_pt_line_quad: The number of point in a line of our quadrant
        :param cam_coord: The dictionary (CameraType: ((i_min, i_max), (j_min, j_max)))
        holding the beginning / ending index  of each quadrant in the big ground_truth
        :param cam_type: The current camera we have
        :param cam_with_center: The camera holding the center
        """
        ((i_min, i_max), (j_min, j_max)) = cam_coord[cam_type]

        # Put the result in the form of a matrix of tuple as the ground_truth
        quadrant_gt = np.zeros((nb_pt_line_quad, nb_pt_line_quad), dtype=tuple)
        for i in range(nb_pt_line_quad):
            for j in range(nb_pt_line_quad):
                quadrant_gt[i, j] = (cam_type, coord_mat[i, j, :])

        # Save the relevant data in the ground_truth by ignoring the intersection according to the rule
        if i_max - i_min > j_max - j_min:
            if j_min > 0:
                ground_truth[i_min:i_max + 1, j_min:j_max + 1] = quadrant_gt[:, 1:]
            else:
                ground_truth[i_min:i_max + 1, j_min:j_max + 1] = quadrant_gt[:, :nb_pt_line_quad - 1]
        else:
            if i_min > 0:
                ground_truth[i_min:i_max + 1, j_min:j_max + 1] = quadrant_gt[1:, :]
            else:
                ground_truth[i_min:i_max + 1, j_min:j_max + 1] = quadrant_gt[:nb_pt_line_quad - 1, :]

        # The center point may not be visible in all camera
        if cam_with_center == cam_type:
            if cam_with_center == CamType.TOP_RIGHT:
                val = quadrant_gt[nb_pt_line_quad - 1, 0]
            elif cam_with_center == CamType.TOP_LEFT:
                val = quadrant_gt[nb_pt_line_quad - 1, nb_pt_line_quad - 1]
            elif cam_with_center == CamType.BOT_RIGHT:
                val = quadrant_gt[0, 0]
            else:
                val = quadrant_gt[0, nb_pt_line_quad - 1]
            ground_truth[MAX_NB_PT_LINE // 2, MAX_NB_PT_LINE // 2] = val

    @staticmethod
    def save_quadrant_mask(cam_type: CamType, quadrant_mask_contour: np.array):
        """
        Save the coordinates defining the mask of a quadrant into a file
        GALVOS_MASK_FILE_[cam_type].txt in the folder files.
        :param cam_type: The camera type
        :param quadrant_mask_contour: List of coordinates defining the mask
        """
        name = GALVOS_MASK_FILE.replace(".txt", f"_{cam_type.value}.txt")
        try:
            os.remove(name)
        except FileNotFoundError:
            pass
        ground_truth_file = open(name, 'w')
        for c in quadrant_mask_contour[0]:
            ground_truth_file.writelines(f"{c[0], c[1]}\n")
        ground_truth_file.close()
        print(f"{bcolors.OKCYAN}Quadrant Mask for {cam_type} saved!")

    @staticmethod
    def load_quadrant_mask(cam_type: CamType):
        """
        Load the quadrant mask from a previous calibration run
        :param cam_type: The camera type of the quadrant we want
        """
        mask_contours = []
        ground_truth_file = open(GALVOS_MASK_FILE.replace(".txt", f"_{cam_type.value}.txt"), 'r')

        for line in ground_truth_file.readlines():
            cnt = line.replace("(", "").replace(")", "").replace("\n", "").replace(" ", "")
            if ',' in line:
                x, y = cnt.split(',')
                mask_contours.append((int(x), int(y)))
        ground_truth_file.close()
        return mask_contours

    @staticmethod
    def extract_sub_ground_truth(full_ground_truth: np.array, nb_pt_on_line: int):
        """
        Create a smaller matrix from the full one of our gt.
        :param full_ground_truth: The MAX_NB_PT_LINExMAX_NB_PT_LINEx2 matrix of our points
        :param nb_pt_on_line: The desired number of points on a line. Values need to be contained
        in NB_POINT_PER_CALIB values to work
        """
        if nb_pt_on_line == MAX_NB_PT_LINE:
            return full_ground_truth

        if nb_pt_on_line not in [math.sqrt(k) for k in NB_POINT_PER_CALIB.values()]:
            raise ArgumentException("Number of line chosen incorrect")

        step = math.ceil((MAX_NB_PT_LINE - 1) / (nb_pt_on_line - 1))
        return full_ground_truth[0:MAX_NB_PT_LINE:step, 0:MAX_NB_PT_LINE:step]
