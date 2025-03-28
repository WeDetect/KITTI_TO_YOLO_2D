import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from const import CLASS_NAME_TO_ID, RELATIVE_PATH_TO_LABEL_2, RELATIVE_PATH_TO_CALIB, RELATIVE_PATH_TO_VELODYNE
from src.kitti_util import Calibration  # and any other functions you need from here


def point_cloud_2_top(points, res=0.1, zres=0.3,
                      side_range=(-20., 20 - 0.05),  # left-most to right-most in meters
                      fwd_range=(0., 40. - 0.05),  # back-most to forward-most in meters
                      height_range=(-2., 0.)):  # bottom-most to upper-most in meters
    """
    Create a bird's eye view (BEV) representation of the point cloud.

    Parameters
    ----------
    points : numpy.ndarray
        Array of points (N x 4) with x, y, z and reflectance.
    res : float
        Resolution (meters) of each pixel.
    zres : float
        Resolution in the z-axis.
    side_range : tuple
        (min, max) lateral limits (in meters).
    fwd_range : tuple
        (min, max) forward limits (in meters).
    height_range : tuple
        (min, max) height limits (in meters).

    Returns
    -------
    top : numpy.ndarray
        BEV image with dimensions corresponding to (y_max+1, x_max+1, z_max+1).
    """
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    reflectance = points[:, 3]

    # Determine the image dimensions.
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    z_max = int((height_range[1] - height_range[0]) / zres)
    top = np.zeros((y_max + 1, x_max + 1, z_max + 1), dtype=np.float32)

    # Create a filter for points inside the desired x and y ranges.
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filt = np.logical_and(f_filt, s_filt)

    # Iterate over height slices.
    for i, height in enumerate(np.arange(height_range[0], height_range[1], zres)):
        z_filt = np.logical_and(z_points >= height, z_points < height + zres)
        zfilter = np.logical_and(filt, z_filt)
        indices = np.argwhere(zfilter).flatten()

        xi_points = x_points[indices]
        yi_points = y_points[indices]
        zi_points = z_points[indices]
        ref_i = reflectance[indices]

        # Convert to pixel positions (swap axes as needed).
        x_img = (-yi_points / res).astype(np.int32)
        y_img = (-xi_points / res).astype(np.int32)

        # Shift pixels to ensure (0, 0) is at the minimum.
        x_img -= int(np.floor(side_range[0] / res))
        y_img += int(np.floor(fwd_range[1] / res))

        # Compute pixel intensity (height values normalized relative to height_range).
        pixel_values = zi_points - height_range[0]
        top[y_img, x_img, i] = pixel_values

        # Store reflectance values in the last channel.
        top[y_img, x_img, z_max] = ref_i

    # Normalize and scale to 8-bit values.
    top = (top / np.max(top) * 255).astype(np.uint8)
    return top


def transform_to_img(xmin, xmax, ymin, ymax, res=0.1,
                     side_range=(-20., 20 - 0.05),
                     fwd_range=(0., 40. - 0.05)):
    """
    Transform coordinates to image pixel indices.

    Returns
    -------
    tuple of floats
        (xmin_img, xmax_img, ymin_img, ymax_img)
    """
    xmin_img = -ymax / res - side_range[0] / res
    xmax_img = -ymin / res - side_range[0] / res
    ymin_img = -xmax / res + fwd_range[1] / res
    ymax_img = -xmin / res + fwd_range[1] / res

    return xmin_img, xmax_img, ymin_img, ymax_img


def draw_point_cloud(ax, points, axes=[0, 1, 2], point_size=0.1,
                     xlim3d=None, ylim3d=None, zlim3d=None):
    """
    Draw a point cloud on the provided axis.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to draw on.
    points : numpy.ndarray
        Array of points.
    axes : list
        Which axes to project (default: [0,1,2]).
    point_size : float
        Size of the points.
    xlim3d, ylim3d, zlim3d : tuple or None
        Optional limits for 3D plots.
    """
    axes_limits = [
        [-20, 80],  # X-axis limits
        [-40, 40],  # Y-axis limits
        [-3, 3]  # Z-axis limits
    ]
    axes_str = ['X', 'Y', 'Z']
    ax.grid(False)
    ax.scatter(*np.transpose(points[:, axes]), s=point_size, c=points[:, 3], cmap='gray')
    ax.set_xlabel(f"{axes_str[axes[0]]} axis")
    ax.set_ylabel(f"{axes_str[axes[1]]} axis")
    if len(axes) > 2:
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_zlabel(f"{axes_str[axes[2]]} axis")
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])

    if xlim3d is not None:
        ax.set_xlim3d(xlim3d)
    if ylim3d is not None:
        ax.set_ylim3d(ylim3d)
    if zlim3d is not None:
        ax.set_zlim3d(zlim3d)


def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    """
    Compute the 3D bounding box in camera (cam2) coordinates.

    Parameters
    ----------
    h, w, l : float
        Dimensions of the box.
    x, y, z : float
        Center location.
    yaw : float
        Yaw angle (rotation around the y-axis).

    Returns
    -------
    numpy.ndarray
        3 x 8 array of corner coordinates in cam2 coordinate system.
    """
    R = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2


def draw_box(ax, vertices, axes=[0, 1, 2], color='black'):
    """
    Draw a 3D bounding box on the provided axis.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to draw on.
    vertices : numpy.ndarray
        8 x 3 array of box vertices.
    axes : list, optional
        Axes order to plot (default is [0, 1, 2]).
    color : str, optional
        Color of the box lines.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # lower plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # upper plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # vertical connections
    ]
    for connection in connections:
        ax.plot(*vertices[:, connection], c=color, lw=0.5)


def read_detection(path):
    """
    Read a KITTI detection label file.

    Parameters
    ----------
    path : str
        Path to the detection file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with appropriate columns.
    """
    df = pd.read_csv(path, header=None, sep=' ')
    df.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                  'bbox_right', 'bbox_bottom', 'height', 'width', 'length',
                  'pos_x', 'pos_y', 'pos_z', 'rot_y']
    df.reset_index(drop=True, inplace=True)
    return df


def load_calibration_and_point_cloud(image_id, base_path):
    """
    Load calibration and point cloud data for a given image.

    Parameters
    ----------
    image_id : int
        Image identifier.
    base_path : str
        Base path to the KITTI dataset.

    Returns
    -------
    tuple
        Calibration object and point cloud array.
    """
    calib_path = os.path.join(base_path, RELATIVE_PATH_TO_CALIB, f"{image_id:06d}.txt")
    velodyne_path = os.path.join(base_path, RELATIVE_PATH_TO_VELODYNE, f"{image_id:06d}.bin")
    calib = Calibration(calib_path)
    points = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
    return calib, points


def load_labels(image_id, base_path):
    """
    Load detection labels for a given image.

    Parameters
    ----------
    image_id : int
        Image identifier.
    base_path : str
        Base path to the KITTI dataset.

    Returns
    -------
    pandas.DataFrame
        DataFrame of detection labels.
    """
    label_2_path = os.path.join(base_path, RELATIVE_PATH_TO_LABEL_2, f"{image_id:06d}.txt")
    return read_detection(label_2_path)


def plot_bev_image(points, df, calib, draw_boxes):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Generate the BEV image.
    top = point_cloud_2_top(points, zres=1.0,
                            side_range=(-40., 40 - 0.05),
                            fwd_range=(0., 80. - 0.05))
    top = np.array(top, dtype=np.float32)
    ax.imshow(top, aspect='equal')

    # Retrieve the BEV image dimensions for normalization.
    img_height, img_width = top.shape[0], top.shape[1]

    yolo_labels = []
    for o in range(len(df)):
        obj_class = df.loc[o, 'type']
        class_id = CLASS_NAME_TO_ID.get(obj_class, -1)
        if class_id == -1:
            continue

        corners_3d_cam2 = compute_3d_box_cam2(
            *df.loc[o, ['height', 'width', 'length',
                        'pos_x', 'pos_y', 'pos_z', 'rot_y']])
        corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)

        if draw_boxes:
            draw_3d_bounding_box(ax, corners_3d_velo)

        # Pass image dimensions so the labels are normalized.
        yolo_label = convert_to_yolo_format(class_id, corners_3d_velo,
                                            res=0.1,
                                            side_range=(-40., 40 - 0.05),
                                            fwd_range=(0., 80. - 0.05),
                                            img_size=(img_width, img_height))
        yolo_labels.append(yolo_label)

    plt.axis('off')
    plt.tight_layout()
    return fig, ax, yolo_labels


def draw_3d_bounding_box(ax, corners_3d_velo):
    """
    Draw a 3D bounding box on a given axis using the transformed corner coordinates.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to draw on.
    corners_3d_velo : numpy.ndarray
        3D corners in Velodyne coordinates.
    """
    x1, x2, x3, x4 = corners_3d_velo[0:4, 0]
    y1, y2, y3, y4 = corners_3d_velo[0:4, 1]
    x1, x2, y1, y2 = transform_to_img(x1, x2, y1, y2,
                                      side_range=(-40., 40 - 0.05),
                                      fwd_range=(0., 80. - 0.05))
    x3, x4, y3, y4 = transform_to_img(x3, x4, y3, y4,
                                      side_range=(-40., 40 - 0.05),
                                      fwd_range=(0., 80. - 0.05))
    lines = [
        [(x1, y1), (x2, y2)],
        [(x2, y2), (x3, y3)],
        [(x3, y3), (x4, y4)],
        [(x4, y4), (x1, y1)]
    ]
    for line in lines:
        xs, ys = zip(*line)
        ax.add_line(Line2D(xs, ys, linewidth=1, color='red'))


def convert_to_yolo_format(class_index, corners_3d_velo, res=0.1,
                           side_range=(-40., 40 - 0.05),
                           fwd_range=(0., 80. - 0.05),
                           img_size=None):
    """
    Convert a 3D bounding box (in Velodyne coordinates) to YOLO format in BEV image space.

    Parameters
    ----------
    class_index : int
        The class index.
    corners_3d_velo : numpy.ndarray
        Array of 3D bounding box corners (at least 4 corners) in Velodyne coordinates.
    res : float, optional
        Resolution used in BEV conversion.
    side_range : tuple, optional
        Lateral limits used in BEV conversion.
    fwd_range : tuple, optional
        Forward limits used in BEV conversion.
    img_size : tuple or None, optional
        (img_width, img_height) of the BEV image. If provided, output coordinates will be normalized.

    Returns
    -------
    str
        YOLO formatted label: "<class_index> <x_center> <y_center> <width> <height>"
    """
    # Extract the first 4 corners (assuming these define the base of the box)
    x1, x2, x3, x4 = corners_3d_velo[0:4, 0]
    y1, y2, y3, y4 = corners_3d_velo[0:4, 1]

    # Transform the corners from Velodyne coordinates to BEV image coordinates.
    x1, x2, y1, y2 = transform_to_img(x1, x2, y1, y2, res=res,
                                      side_range=side_range, fwd_range=fwd_range)
    x3, x4, y3, y4 = transform_to_img(x3, x4, y3, y4, res=res,
                                      side_range=side_range, fwd_range=fwd_range)

    # Compute bounding box in pixel coordinates.
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    # Normalize the coordinates if image size is provided.
    if img_size is not None:
        img_width, img_height = img_size
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

    return f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def normalize_yolo_bbox(x, y, w, h, img_size=640):
    """
    Normalize bounding box parameters for YOLO training.

    Parameters
    ----------
    x, y, w, h : float
        Bounding box center coordinates and dimensions in pixel units.
    img_size : int, optional
        Assumed image size (default is 640). Adjust as necessary.

    Returns
    -------
    tuple
        Normalized (x, y, w, h) values between 0 and 1.
    """
    return x / img_size, y / img_size, w / img_size, h / img_size


def save_bev_image(fig, save_path):
    """
    Save the BEV image to disk.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the BEV image.
    save_path : str
        Path where the image will be saved.
    """
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    print(f"BEV image saved to {save_path}")


def save_yolo_labels(yolo_labels, label_path):
    """
    Save YOLO labels to disk.

    Parameters
    ----------
    yolo_labels : list of str
        List of YOLO formatted labels.
    label_path : str
        Path where the label file will be saved.
    """
    with open(label_path, 'w') as f:
        f.write("\n".join(yolo_labels))
    print(f"YOLO labels saved to {label_path}")


def convert_point_cloud_to_bev(image_id, base_path, save_path=None, label_path=None, draw_boxes=False):
    """
    Process a single image: load data, generate BEV image, and save labels.

    Parameters
    ----------
    image_id : int
        Image identifier.
    base_path : str
        Base path to the KITTI dataset.
    save_path : str or None
        Path to save the BEV image. If None, image is not saved.
    label_path : str or None
        Path to save YOLO labels. If None, labels are not saved.
    draw_boxes: boolean, default False
        Draws bboxes on the bev image
    """
    calib, points = load_calibration_and_point_cloud(image_id, base_path)
    df = load_labels(image_id, base_path)
    fig, ax, yolo_labels = plot_bev_image(points, df, calib, draw_boxes)

    if save_path:
        save_bev_image(fig, save_path)
    if label_path:
        save_yolo_labels(yolo_labels, label_path)
