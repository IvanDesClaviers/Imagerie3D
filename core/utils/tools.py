import cv2
import math
import rtree
import numpy as np


def create_extrinsic(rot, trans):
    """
    Create a square extrinsic matrix from the rotation and translation matrix.
    :param rot: The 3x3 rotation matrix.
    :param trans: the 3x1 translation matrix.
    :return: Kext the 4x4 extrinsic matrix.
    """
    kext = np.zeros([4, 4], dtype=np.float32)
    kext[0:3, 0:3] = rot
    kext[0:3, 3] = trans[:].T
    kext[3, 3] = 1
    return kext


def moments(c):
    """
    Get the logical center of a shape
    :param c: The array from cv2.findContours
    :return: A tuple with the coordinates
    """
    m = cv2.moments(c)
    if m['m00'] != 0.0:
        return int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])
    else:
        c = c[:, 0, :]
        return ((c[:, 0].min() + c[:, 0].max()) // 2,
                (c[:, 1].min() + c[:, 1].max()) // 2)


def dist_2d(p1: tuple, p2: tuple):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def order_points(img, points, verbose=0, is_reverse=False, min_h=50):
    """
    Order a point struct so the coordinates are ordered from top to bottom, left to right
    :param img: Image for showing the debugging result
    :param points: Nx1x2 Points list to sort
    :param verbose: Show the rows created on an image
    :param is_reverse: Right to left if true instead
    :param min_h: minimum height between 2 point
    :return: The Nx1x2 sorted Points list
    """

    sorted_points = []
    it = 0
    while len(points) > 0 and it < 1000:
        sum_coord = {(points[k, 0, 0], points[k, 0, 1])
                     : points[k, 0, 0] + points[k, 0, 1] for k in range(len(points))}
        sum_coord_sorted = sorted(sum_coord.items(), key=lambda item: item[1])
        up_left = sum_coord_sorted[0][0]  # find upper left point

        sum_coord = {(points[k, 0, 0], points[k, 0, 1])
                     : points[k, 0, 0] - points[k, 0, 1] for k in range(len(points))}
        sum_coord_sorted = sorted(sum_coord.items(), key=lambda item: item[1])
        up_right = sum_coord_sorted[-1][0]  # find upper right point

        # convert opencv keypoint to numpy 3d point

        a = np.array([up_left[0], up_left[1], 0])
        b = np.array([up_right[0], up_right[1], 0])

        row_points = []
        remaining_points = []
        for k in points[:, 0, :]:
            p = np.array([k[0], k[1], 0])
            # distance between keypoint and line a->b
            dist = np.linalg.norm(np.cross(np.subtract(p, a), np.subtract(b, a))) / np.linalg.norm(b)
            if min_h > dist:
                row_points.append(k)
            else:
                remaining_points.append(k)

        sorted_row = sorted(row_points, key=lambda h: h[0], reverse=is_reverse)

        sorted_points.extend(sorted_row)
        points = np.zeros((len(remaining_points), points.shape[1], points.shape[2]),
                          dtype=np.float32)
        if verbose > 3:
            cv2.line(img, (int(sorted_row[0][0]), int(sorted_row[0][1])),
                     (int(sorted_row[-1][0]), int(sorted_row[-1][1])), (0, 0, 255), 3)

        for k in range(len(points)):
            points[k] = remaining_points[k]
        it += 1
    if it == 1000:
        print("WARNING: Point ordering did not work Correctly")
    # Reshape them the correct way
    points = np.zeros((len(sorted_points), points.shape[1], points.shape[2]),
                      dtype=np.float32)
    if verbose > 2:
        from core.utils import show
        show(img, "Ordered dots (All dots should be linked in line. Press any key to continue)")
    for k in range(len(sorted_points)):
        points[k] = sorted_points[k]
    if verbose > 3:
        for pt in points:
            cv2.circle(img, tuple(pt[0]), 10, (0, 255, 0))
    return points


def remove_close_points(points, min_dist):
    """Return a maximal list of elements of points such that no pairs of
    points in the result have distance less than r.
    Note: Complexity is nLog(n), not mine
    :param points: List of 2d points coordinates
    :param min_dist: The minimum distance in the same unit as the coordinates
    """
    result = []
    index = rtree.index.Index()
    for i, p in enumerate(points):
        px, py = p
        nearby = index.intersection((px - min_dist, py - min_dist, px + min_dist, py + min_dist))
        if all(dist_2d(p, points[j]) >= min_dist for j in nearby):
            result.append(p)
            index.insert(i, (px, py, px, py))
    return result


def get_intrinsic_from_datasheet(img_shape, f, sensor_size):
    """
    Get the intrinsic camera parameter from its information
    :param img_shape: The tuple (width, height) in pixel
    :param f: The tuple (fx, fy) focal length in [m]
    :param sensor_size: The tuple (sx, sy) sensor size for width and height in [m]
    :return:
    """
    return np.array([[f[0] * img_shape[0] / sensor_size[0], 0.0, img_shape[0] / 2],
                     [0.0, f[1] * img_shape[1] / sensor_size[1], img_shape[1] / 2],
                     [0.0, 0.0, 1.0]])


def order_point_custom(points: list, max_delta_Y: int = 10, show_img=False, frame_shape=(1920, 1080, 3)):
    """
    This algorithm will sort a cloud points to have a list of list with points
    with a minimum delta Y separating them
    :param points: A list of 2D points to sort
    :param max_delta_Y: The distance between 2 points in Y in [pixel]
    :param show_img: True to show the data ordered with random colors
    :param frame_shape: The shape of the image if you want to redefine it
    :return: A list of line from top to bottom, a line being a list of point ordered left to right
    """
    line_idx = 0
    points = list(filter(lambda a: a != (0., 0.), points))
    # Setup for first loop
    points = sorted(points, key=lambda x: x[1])  # order them by Y
    up_left = sorted(points, key=lambda p: p[0] + p[1])[0]  # find upper right point
    ordered_points = [[up_left]]
    points.remove(up_left)
    while 0 < len(points):
        candidate = None
        min_dx = math.inf
        k = 0
        # Whether we found a very good candidate, or we went too far we are sure we are on another line
        while min_dx > 50 * (k // 40 + 1) and k < len(points):
            pt = points[k]
            if abs(pt[1] - up_left[1]) <= max_delta_Y:  # We are on our line
                dx = pt[0] - up_left[0]
                if max_delta_Y < dx < min_dx:  # We found a point closer in x than the others
                    candidate = pt
                    min_dx = dx
            k += 1

        if min_dx != math.inf and candidate in points:
            ordered_points[line_idx].append(candidate)
            points.remove(candidate)
            up_left = candidate
        else:  # no right neighbour left
            line_idx += 1
            # find upper left
            sorted_pts = sorted(points, key=lambda p: (p[0]) + (p[1]))
            if len(sorted_pts) > 1:
                up_left = sorted_pts[0]
            else:
                break
            ordered_points.append([up_left])
            points.remove(up_left)

    for k in range(len(ordered_points)):
        ordered_points[k] = sorted(ordered_points[k], key=lambda x: x[0])  # sort them by x

    if show_img:
        from core.utils.data_shower import show_grid_intersections
        show_grid_intersections(ordered_points, frame_shape)

    return ordered_points


def get_vect_delta(vect: np.array, first: int = None, last: int = None):
    """
    Compute the delta of a vector of points. It is the mean of the difference between consecutive terms of the vector
    :param vect: a Nx2 vector
    :param first: The starting index for the delta, 0 by default
    :param last: The end index for the delta, N by default
    :return: The list [delta_X, delta_Y]
    """
    first = 0 if first is None else first
    last = len(vect) if last is None else last
    if first >= last:
        raise IndexError("_get_line_delta: Starting index must be lower than end index")
    all_dx = [(a[0] - b[0]) for a, b in zip(vect[1:], vect[:len(vect) - 1])]
    all_dy = [(a[1] - b[1]) for a, b in zip(vect[1:], vect[:len(vect) - 1])]
    return [np.mean(all_dx[first:last]), np.mean(all_dy[first:last])]
