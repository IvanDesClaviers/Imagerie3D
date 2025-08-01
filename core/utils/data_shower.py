import cv2
import random
import matplotlib.pyplot as plt
from math import asin, pi, sqrt

from core.stereo_calibration.config import *


def show(image, name, resize=True, block=False, norm=False):
    """
    A show function using opencv to avoid 4 lines of codes everytime
    """
    if norm and len(image.shape) > 2:
        from core.utils import equalize_image
        img = equalize_image(image)
    else:
        img = image
    if resize:
        cv2.namedWindow(name, cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(name, 860, 640)

    cv2.imshow(name, img)
    if block:
        cv2.waitKey()


def compute_calib_error(real_points, img_points, intrinsic, rot, trans, dist):
    """
    Compute the stereo_calibration re-projection error
    :param real_points: Object point used
    :param img_points: Image points used
    :param intrinsic: The 3x3 intrinsic matrix
    :param rot: The Nx3x3 rotation matrix
    :param trans:he Nx3x3 translation matrix
    :param dist: The 1x5 Distortion coefficients
    """
    mean_error = 0
    for i in range(len(real_points)):
        img_points2, _ = cv2.projectPoints(real_points[i], rot[i], trans[i], intrinsic, dist)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        mean_error += error
        print(error)
    print()
    print(f"Total error: {mean_error / len(real_points)} pixels")

    if mean_error / len(real_points) > 2:
        raise Exception("Error is too high, some pictures may be flawed. "
                        "Remove pictures with too high error")


def compare_galvos_extrinsic(k_cam_galv):
    """
    Show where the stereo stereo_calibration of the projector places the projector compared
    to where it should have been
    :param k_cam_galv: Extrinsic parameters from the stereo-stereo_calibration
    """
    from core import apply_extrinsic
    origin = [0, 0, 0]
    kext = k_cam_galv.copy()
    galvos = apply_extrinsic([0, 0, 0], np.linalg.inv(kext))

    ideal_zone = np.array([[GALVOS_ZONE[0][0], GALVOS_ZONE[0][0], GALVOS_ZONE[0][0], GALVOS_ZONE[0][0],
                            GALVOS_ZONE[0][1], GALVOS_ZONE[0][1], GALVOS_ZONE[0][1], GALVOS_ZONE[0][1]],
                           [GALVOS_ZONE[1][0], GALVOS_ZONE[1][0], GALVOS_ZONE[1][1], GALVOS_ZONE[1][1],
                            GALVOS_ZONE[1][0], GALVOS_ZONE[1][0], GALVOS_ZONE[1][1], GALVOS_ZONE[1][1]],
                           [GALVOS_ZONE[2][0], GALVOS_ZONE[2][1], GALVOS_ZONE[2][0], GALVOS_ZONE[2][1],
                            GALVOS_ZONE[2][0], GALVOS_ZONE[2][1], GALVOS_ZONE[2][0], GALVOS_ZONE[2][1]]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(origin[0], origin[1], origin[2], marker='o')
    ax.scatter(galvos[0], galvos[1], galvos[2], marker='^')
    ax.scatter(ideal_zone[0], ideal_zone[1], ideal_zone[2], marker='s')

    # Create cubic bounding box to simulate equal aspect ratio
    x = np.array([origin[0], ideal_zone[0][0], galvos[0]], dtype=np.float32)
    y = np.array([origin[1], ideal_zone[1][0], galvos[1]], dtype=np.float32)
    z = np.array([origin[2], ideal_zone[2][0], galvos[2]], dtype=np.float32)

    max_r = np.array([x.max(initial=None) - x.min(initial=None), y.max(initial=None) - y.min(initial=None),
                      z.max(initial=None) - z.min(initial=None)]).max(initial=None)
    xb = 0.5 * max_r * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max(initial=None) + x.min(initial=None))
    yb = 0.5 * max_r * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max(initial=None) + y.min(initial=None))
    zb = 0.5 * max_r * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max(initial=None) + z.min(initial=None))
    for xb, yb, zb in zip(xb, yb, zb):
        ax.plot([xb], [yb], [zb], 'w')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    if not (min(GALVOS_ZONE[0][0], GALVOS_ZONE[0][1]) < galvos[0] < max(GALVOS_ZONE[0][0], GALVOS_ZONE[0][1]) and
            min(GALVOS_ZONE[1][0], GALVOS_ZONE[1][1]) < galvos[1] < max(GALVOS_ZONE[1][0], GALVOS_ZONE[1][1]) and
            min(GALVOS_ZONE[2][0], GALVOS_ZONE[2][1]) < galvos[2] < max(GALVOS_ZONE[2][0], GALVOS_ZONE[2][1])):
        print("WARNING: The Galvos result position is outside the expected area. "
              "Redo the stereo_calibration with a higher verbose in order to find"
              "problematic pictures and remove them")


def plot_vectors(norm1, norm2, label1='Perfect plan', label2='Real plan', verbose=0):
    """
    Plot and compare plane 2 to plane 1, show their orientation to each other
    :param label1: Label for plotting vec1
    :param label2: Label for plotting vec2
    :param norm1: The first 3x1 vector
    :param norm2: The 2nd 3x1 vector
    :param verbose: Influence quantity of shown data
    """
    # Compute angles difference
    r = np.cross(norm1, norm2)
    angle = asin(sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)) * 180 / pi
    print(f"Angle between the vectors is {angle} deg")
    print()
    print("Angle Around Camera Pitch (X axis):", asin(r[0]) * 180 / pi, "deg")
    print("Angle Around Camera Yaw (Y axis):", asin(r[1]) * 180 / pi, "deg")
    print("Angle Around Camera Roll (Z axis) :", asin(r[2]) * 180 / pi, "deg")

    if verbose > 0:
        print(norm1)
        print(norm2)

    if verbose > 1:
        print("Drawing Vectors: ")

        # plot the surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([0, 1], [0, 0], zs=[0, 0], color='b', label='X')
        ax.plot([0, 0], [0, 1], zs=[0, 0], color='b', label='Y')
        ax.plot([0, 0], [0, 0], zs=[0, 1], color='b', label='Z')

        ax.plot([0, norm1[0]], [0, norm1[1]], zs=[0, norm1[2]], color='m', label=label1)
        ax.plot([0, norm2[0]], [0, norm2[1]], zs=[0, norm2[2]], color='k', label=label2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    if not (-10 < asin(r[0]) * 180 / pi < 10 and
            -10 < asin(r[1]) * 180 / pi < 10 and
            -5 < asin(r[2]) * 180 / pi < 5):
        raise Exception("Found angles are likely too big to represent real case")


def show_clickable(frame: np.array, callback, full_screen=True):
    """
    Display an image and use a callback on it
    :param frame: The image to display
    :param callback: function(event) called whenever the image is clicked
    :param full_screen: True to set the image in full_screen
    :return:
    """
    import matplotlib
    matplotlib.use('TkAgg')

    # Put the image in full screen with no white border
    if full_screen:
        plt.get_current_fig_manager().full_screen_toggle()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # Add the image to the plot
    plt.gca().imshow(frame, aspect='auto', cmap='gray', interpolation='nearest')
    # Setup the interactive window and display it
    cid = plt.gcf().canvas.mpl_connect('button_press_event', callback)
    plt.show()
    # Disconnect the callbacks after the user has finished
    plt.gcf().canvas.mpl_disconnect(cid)


def show_grid_intersections(grid_points, shape, name="line sort"):
    mask = np.zeros(shape, dtype=np.uint8)
    for line in grid_points:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for pt in line:
            cv2.circle(mask, (int(pt[0]), int(pt[1])), 10, color, 5)
    show(mask, name, block=True)
