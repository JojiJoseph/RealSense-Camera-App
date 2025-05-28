import tyro
from utils import rgbd_to_vertex_data
from visualizer import Visualizer

def main(rgb_path: str, depth_path: str, fx: float, fy: float, cx: float, cy: float, output_path: str = "pcd.ply"):
    """
    Main function to visualize RGB-D data and export point cloud.
    :param rgb_path: Path to the RGB image
    :param depth_path: Path to the depth image
    :param fx: Focal length in x direction
    :param fy: Focal length in y direction
    :param cx: Optical center in x direction
    :param cy: Optical center in y direction
    """
    import cv2
    import numpy as np

    # Load RGB and depth images
    color_image = cv2.imread(rgb_path)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    visualizer = Visualizer()
    while True:
        key = visualizer.visualize(color_image, depth_image, fx, fy, cx, cy)

        if key == ord("q"):
            break
    vertices = rgbd_to_vertex_data(
        color_image.astype(float) / 255,
        np.asanyarray(depth_image).astype(float),
        fx,
        fy,
        cx,
        cy,
    )
    from plyfile import PlyData, PlyElement

    vertices[:, 3:6] = vertices[:, 5:2:-1] * 255

    vertices = np.array(
        [tuple(p) for p in vertices],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el], text=True).write(output_path)

if __name__ == "__main__":
    tyro.cli(main)
