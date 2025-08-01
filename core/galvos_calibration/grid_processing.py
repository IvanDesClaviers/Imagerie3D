from global_config import *
from core.utils import *
from core.utils.image_processing import *
from core.utils.tools import get_vect_delta

from core import ComputerVisionException


def find_laser(frame, mask: np.array = None, verbose=0):
    """
    Return the center of a laser shape on a rgb pictures by thresholding it
    :param frame: The rgb picture
    :param mask: An optional binary image for computing the detection in a limited area
    :param verbose: Show threshed images if not null
    :return: The tuple of the coordinate of the laser center
    """
    # Preprocess
    clone = frame.copy()
    green = cv2.bitwise_and(mask, clone[:, :, 1]) if mask is not None else clone[:, :, 1]
    print(f"Max: {green.max()}")

    if green.max() < 30:
        raise ComputerVisionException("Laser Finder: Nothing interesting detected")

    _, thresh = cv2.threshold(green, green.max() - 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise ComputerVisionException("Laser Finder:Imaged preprocessing failed. "
                                      "Try changing your input threshold")

    if len(contours) > 5:
        raise ComputerVisionException("Laser Finder:Too many contours found during preprocess. "
                                      "Binary transformation did not went well")

    # Take the biggest contour (only one)
    contours = sorted(contours, key=len)
    laser_center = moments(contours[-1])
    if verbose > 1:
        cv2.circle(frame, laser_center, 5, (0, 0, 255), 5)
        show(frame, "Laser Center", block=False)

    if verbose > 2:
        show(thresh, "Threshed Image", block=False)

    if verbose > 3:
        show(green, "Green Masked", block=False)

    return laser_center


def find_laser_accurately(frame, mask: np.array = None, verbose=0):
    """

    :param frame: The rgb picture
    :param mask: An optional binary image for computing the detection in a limited area
    :param verbose: Show threshed images if not null
    :return: The tuple of the coordinate of the laser center
    """
    clone = frame.copy()
    green = cv2.bitwise_and(mask, clone[:, :, 1]) if mask is not None else clone[:, :, 1]
    print(f"Max: {green.max()}")
    if green.max() < 30:
        raise ComputerVisionException("Laser Finder: Nothing interesting detected")

    thresh = 3 * (green.max() // 4)
    _, im_i = cv2.threshold(green, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(im_i, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    thresh_img = im_i

    if not len(contours) > 0:
        raise ComputerVisionException("Threshold selected too high or laser not visible")

    nb_loop = 0
    step = green.max() // 20
    # Find the brightest zone of the laser
    while thresh <= green.max():
        _, im_i = cv2.threshold(green, thresh, 255, cv2.THRESH_BINARY)
        temp, _ = cv2.findContours(im_i, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not len(temp) > 0:
            print(f"Number of loop to find optimal center: {nb_loop}")
            break
        contours = temp
        thresh = min(thresh + step, green.max() + 1)
        thresh_img = im_i
        nb_loop += 1
        if verbose > 4:
            show(thresh_img, f"Thresh {thresh}", block=False)

    if verbose > 3:
        cv2.waitKey()
        cv2.destroyAllWindows()

    # We have the brightest pixels
    laser_center = (0, 0)
    for contour in contours:
        a, b = moments(contour)
        laser_center = (laser_center[0] + a, laser_center[1] + b)
    laser_center = (int(laser_center[0] / len(contours)), int(laser_center[1] / len(contours)))

    if verbose > 1:
        cv2.circle(frame, laser_center, 3, (0, 0, 255), 1)
        show(frame, "Laser Center Accurate", block=False)

    if verbose > 2:
        show(thresh_img, "Threshed Image Accurate", block=False)

    if verbose > 3:
        show(green, "Green Masked accurate", block=False)

    return laser_center


def get_calibration_intersections(quadrant_intersections: list, n=41, frame_shape=(1920, 1080, 3), show_img=False):
    """
    Get the matrix of intersection coordinates based on
    :param quadrant_intersections: A list of tuple coordinates
    :param n: The expected number of point on the quadrant
    :param frame_shape: The size of the frame the coordinates where taken
    :param show_img: True to show the result
    :return:
    """
    ordered_lines = order_point_custom(quadrant_intersections, frame_shape=frame_shape, show_img=show_img)
    intersection_matrix = interpolate_missing_intersections(ordered_lines, n, 30)
    try:
        # To ensure the validity of our extrapolation in y coordinates,
        # since our result is a 41x41 matrix, it should be invariant to rotation.
        # Knowing so, we rotate by 90 degree clockwise and redo the algorithm
        # It helps to detect and remove artifacts from bad image processing
        intersection_matrix_inversed = inverse_matrix_first_channels(intersection_matrix)
        res_rot = [(y[0], y[1]) for x in intersection_matrix_inversed for y in x]
        # Count number of valid lines, it will removes points when we reverse it
        ordered_lines_rotated = order_point_custom(res_rot, frame_shape=(frame_shape[1], frame_shape[0], 3),
                                                   show_img=False)
        intersection_matrix_inversed = interpolate_missing_intersections(ordered_lines_rotated, n, 30)
        intersection_matrix = inverse_matrix_first_channels(intersection_matrix_inversed)
    except ComputerVisionException:
        # If it fails, just pass and hope for the best
        pass
    if show_img:
        mask = np.zeros(frame_shape, dtype=np.uint8)
        for k in range(n):
            line = intersection_matrix[k, :]
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for pt in line:
                cv2.circle(mask, (int(pt[0]), int(pt[1])), 10, color, 5)
        show(mask, "line interpolated", block=True)
    return intersection_matrix


def get_all_grid_intersections(frame: np.array, show_img: bool = False):
    """
    Process the grid image to extract all cross intersections
    :param frame: The frame of the grid
    :param show_img: Boolean to show intermediate steps
    :return: The image with all intersection
    """

    bright_frame = equalize_image(frame.copy())
    undistorded = undistort_grid_img(bright_frame, show_img=show_img)

    gray = cv2.cvtColor(undistorded, cv2.COLOR_BGR2GRAY)
    clahefilter = cv2.createCLAHE(clipLimit=10000.0, tileGridSize=(21, 21))
    no_glare = clahefilter.apply(gray)

    blur = cv2.blur(no_glare, (13, 13))

    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 55, -2)
    cross = np.copy(bw)
    cross_struct_big = cv2.getStructuringElement(cv2.MORPH_CROSS, (11, 11))
    cross = cv2.erode(cross, cross_struct_big, iterations=2, borderType=cv2.BORDER_REFLECT)
    cross = cv2.dilate(cross, cross_struct_big, iterations=1, borderType=cv2.BORDER_REFLECT)
    cross = cv2.erode(cross, create_losange_morph((11, 11)), iterations=1, borderType=cv2.BORDER_REFLECT)
    intersection_img = cv2.erode(cross, create_losange_morph((3, 3)), iterations=1, borderType=cv2.BORDER_REFLECT)

    if show_img:
        show(intersection_img, "cross", block=True)

    intersections = get_all_center_coord(intersection_img)

    return intersection_img, intersections


def get_all_center_coord(intersection: np.array, too_close_criteria=70):
    """
    Extract the intersections of a binary image of points
    :param intersection: The binary image
    :param too_close_criteria: Remove all points too close from each other
    :return: A list of all eligible coordinates
    """
    cntr, _ = cv2.findContours(intersection, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    intersection_coord = []
    all_perimeters = sorted([cv2.arcLength(c, True) for c in cntr])
    mean_peri = np.mean(all_perimeters[len(all_perimeters) // 4:len(all_perimeters) * 3 // 4])
    for c in cntr:
        peri = cv2.arcLength(c, True)  # Eliminate out-liner contours by size
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)  # Eliminate out-liner contours by shape
        if mean_peri * 0.5 < peri < mean_peri * 2 and 4 == len(approx):
            intersection_coord.append(moments(c))

    intersection_coord = remove_close_points(intersection_coord, too_close_criteria)
    intersection_coord = sorted(intersection_coord, key=lambda x: x[1])  # order them by Y

    print(f"Finished! Found {len(intersection_coord)} valid centers from {len(cntr)} detected!")
    return intersection_coord


def delimit_quadrant(img: np.array, intersections, write: bool = False):
    """
    Define the boundaries of the quadrant manually to extract the coordinates and the image inside
    :param img: The image used for the selection
    :param intersections: The list of all intersections
    :param write: Option to save the quadrant image
    :return: The quadrant image and
    """
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    mask_coord = []

    def select_grid_border(event):
        if event.button == 1:  # Left click
            mask_coord.append([int(event.xdata), int(event.ydata)])
            print(f"[{int(event.xdata)}, {int(event.ydata)}],")

    show_clickable(img, select_grid_border, full_screen=False)

    cv2.fillPoly(mask, [np.array(mask_coord)], 255)
    quadrant_img = cv2.bitwise_and(mask, img)

    if write:
        cv2.imwrite(os.path.join(IMG_DIR, "quadrant_intersections.png"), quadrant_img)

    quadrant_intersections = []
    for pt in intersections:
        if quadrant_img[pt[1], pt[0]] != 0:
            quadrant_intersections.append(pt)

    return quadrant_img, quadrant_intersections


def interpolate_missing_intersections(lines: list, n: int, same_column_criteria=20):
    """
    Convert a list of list of point to a matrix of size NxNx2. It will interpolate and
    extrapolate missing points by assuming the distance between each adjacent point is in average constant.
    In order to work, we assume WE DON'T MISS TOO MANY POINTS PER LINE and DON'T HAVE WRONG COORDINATES

    :param lines: A list of  nb_pt_on_line lists that should contain nb_pt_on_line lines
    :param n: The expected number of points per line
    :param same_column_criteria: The distance in x to consider 2 points, one above the other, to be on the same column
    :return: The NxNx2 matrix with no missing data
    """
    mat = np.zeros((n, n, 2))
    first_x_value = _extrapolate_first_x_coord(lines, n, same_column_criteria)
    id_line = 0

    for line in lines:
        line = np.array(sorted(line, key=lambda u: u[0]))  # order them by X
        if len(line) == n:  # Nothing to do, all points where correctly detected
            mat[id_line, :] = line
            id_line += 1
        elif n - 20 > len(line):  # Ignore too small lines
            pass
        elif n < len(line):
            # Shouldn't have too much point
            raise ComputerVisionException("Too much point on line, the preprocess must have failed")
        elif len(line) < n:
            j = 1
            line_dx, _ = get_vect_delta(line)

            if id_line == 0:
                previous_x = first_x_value
            elif id_line == 1:
                previous_x = mat[0, 0, 0]
            else:
                previous_x = mat[id_line - 1, 0, 0] + \
                             get_vect_delta(mat[:, 0], max(0, id_line - 4), id_line - 1)[0]

            if abs(line[0][0] - previous_x) < same_column_criteria:  # We have the first element
                mat[id_line, 0] = line[0]
            else:  # first elements are extrapolated
                dx, dy = get_vect_delta(line, 0, 5)
                nb_interpol = _interpolate_points(mat, [previous_x, line[0][1] - dy], line[0], line_dx, id_line, j)
                mat[id_line, 0] = [max(previous_x, mat[id_line, 1, 0] - dx),
                                   max(mat[id_line, 1, 1] - dy, line[0][1] - dy * (nb_interpol + 1))]
                mat[id_line, j + nb_interpol] = line[0]
                j += 1 + nb_interpol

            for k in range(1, len(line)):
                if 1.4 * line_dx < abs(line[k][0] - line[k - 1][0]):  # Add intermediate point if needed
                    j += _interpolate_points(mat, line[k - 1], line[k], line_dx, id_line, j)
                if j >= n:
                    print(f"Something went wrong on line {id_line}, originally had {len(line)} pts.")
                    break

                mat[id_line, j] = line[k]
                j += 1

            if (mat[id_line, :, 0] == 0.).any():  # Extrapolate last points
                _extrapolate_last_points(mat, line, n, id_line)
            id_line += 1
    return mat


def _interpolate_points(mat: np.array, p1: list, p2: list, dx, i: int, j: int):
    """
    Interpolate several points between p1 and p2 for every step of the horizontal distance dx and place them
    on the matrix m at starting from index mat(i,j). It will also add p2 but not p1
    :param mat: The matrix which will have the coordinated edited
    :param p1: First point
    :param p2: Second point
    :param dx: Horizontal step
    :param i: Line index
    :param j: Column index
    :return: The number of point
    """
    nb_interpol = round(abs(p2[0] - p1[0]) / dx) - 1
    for n in range(1, nb_interpol + 1):
        interpolated = [p1[0] + n * (p2[0] - p1[0]) // (1 + nb_interpol),
                        p1[1] + n * (p2[1] - p1[1]) // (1 + nb_interpol)]
        mat[i, j + n - 1] = interpolated
    return nb_interpol


def _extrapolate_first_x_coord(lines: np.array, n: int, same_column_criteria: int):
    """
    Extrapolate the first coordinate in x of a list of lines
    :param lines: Or list of lines
    :param n: The expected number of point in a line
    :return: The extrapolated x value
    """
    certified_first_point = np.zeros((len(lines), 2))
    first_points = np.zeros((len(lines), 2))
    for k in range(len(lines)):
        line = lines[k]
        if n - 20 < len(line):  # Ignore tooo small line
            first_points[k] = line[0]
            if n == len(line):
                certified_first_point[k] = line[0]

    non_zero_indexes = np.where(certified_first_point[:, 0] != 0.)[0]
    if len(non_zero_indexes) > 0:
        non_zero = non_zero_indexes[0]
        return certified_first_point[non_zero][0] + get_vect_delta(certified_first_point, non_zero,
                                                                   min(non_zero + 4, n))[0]
    else:  # There are no full lines
        min_x = np.min([x for x in first_points[:, 0]])
        probable_first_points = np.zeros((len(lines), 2))
        first_point_id = None
        m = -1
        for k in range(len(lines)):
            pt = lines[k][0]
            if pt[0] - min_x < same_column_criteria:
                probable_first_points[m] = pt
                first_point_id = k if first_point_id is None else first_point_id
                m += 1
        if probable_first_points[1][0] == 0:
            return probable_first_points[0][0]
        return probable_first_points[0][0] + get_vect_delta(probable_first_points, 1,
                                                            min(first_point_id + 4, m))[0]


def _extrapolate_last_points(mat: np.array, line: np.array, n: int, i: int):
    """
    Extrapolate the last points of a line and put them on the matrix mat
    :param mat: The matrix to put the value in
    :param line: The line we are processing
    :param n: The expected size of the line
    :param i: The corresponding index of the line on our matrix
    :return:
    """
    first_zero_res = np.where(mat[i, :, 0] == 0.)[0][0]
    first_zero_line = len(line) - (n - first_zero_res)
    for s, r in zip(range(first_zero_res, n), range(first_zero_line, len(line))):
        line_dx, _ = get_vect_delta(mat[i, :s], max(0, s - n // 10), s)
        _, line_dy = get_vect_delta(line, max(0, r - n // 10), r)
        mat[i, s] = [mat[i, s - 1, 0] + line_dx, mat[i, s - 1, 1] + line_dy]
