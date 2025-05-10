import numpy as np
import cv2
from camera import RelsenseCamera
from visualizer import Visualizer
from utils import save_image_pair

camera = RelsenseCamera()


visualizer = Visualizer()


try:
    while True:
        color_image, depth_image = camera.get_frames()
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(np.asanyarray(depth_image), alpha=0.03),
            cv2.COLORMAP_INFERNO,
        )

        fx, fy, cx, cy = camera.get_intrinsics()

        key = visualizer.visualize(color_image, depth_image, fx, fy, cx, cy)
        if key == ord("s"):
            save_image_pair(color_image, depth_image)
        if key == ord("q"):
            break
finally:
    camera.stop()
    cv2.destroyAllWindows()
