import numpy as np
import cv2
from datetime import datetime
import os

def rs_reset(rs):
    """
    Reset the RealSense camera.
    :return: None
    """
    rs_ctx = rs.context()
    devices = rs_ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()


def rgbd_to_vertex_data(rgb, depth, fx, fy, cx, cy, scale_factor=1000.0):
    """
    Convert RGB-D image to vertex data.
    :param rgb: RGB image
    :param depth: Depth image
    :param fx: Focal length in x direction
    :param fy: Focal length in y direction
    :param cx: Optical center in x direction
    :param cy: Optical center in y direction
    :return: Vertex array (N, 6)
    """
    h, w = depth.shape

    # Create a meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Compute the z-coordinates (depth values)
    z = depth.flatten() / scale_factor

    # Compute the x and y coordinates using the intrinsic parameters
    x = (u.flatten() - cx) * z / fx
    y = (v.flatten() - cy) * z / fy
    r = rgb[:, :, 0].flatten()
    g = rgb[:, :, 1].flatten()
    b = rgb[:, :, 2].flatten()

    # Stack the coordinates into a vertex array
    vertices = np.stack((x, y, z, r, g, b), axis=-1)

    return vertices


def opencv_to_opengl_projection(K, viewmat, width, height, near, far):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    proj = np.zeros((4, 4), dtype=np.float32)

    proj[0, 0] = 2 * fx / width
    proj[1, 1] = 2 * fy / height
    proj[0, 2] = 2 * (cx / width) - 1
    proj[1, 2] = 2 * (cy / height) - 1
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2 * far * near / (far - near)
    proj[3, 2] = -1

    inv_arr = np.eye(4)
    inv_arr[1, 1] = -1
    inv_arr[2, 2] = -1
    viewmat = np.dot(inv_arr, viewmat)

    proj = np.dot(proj, viewmat)

    return proj.T  # Transpose for column-major OpenGL

def save_image_pair(color_image, depth_image):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # YearMonthDay_HourMinuteSecond_Millisecond
    os.makedirs("./images", exist_ok=True)
    rgb_path = f"./images/rgb_{timestamp}.png"
    depth_path = f"./images/depth_{timestamp}.png"
    cv2.imwrite(rgb_path, color_image)
    cv2.imwrite(depth_path, depth_image)
    print(f"Saved image pair with timestamp: {timestamp}")
