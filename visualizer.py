import numpy as np
import cv2
import moderngl
from scipy.spatial.transform import Rotation as scipy_rot
from utils import rgbd_to_vertex_data, opencv_to_opengl_projection
import pyrealsense2 as rs

class Visualizer:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(
            vertex_shader=open("./camera.vs.glsl").read(),
            fragment_shader=open("./camera.fs.glsl").read(),
        )
        # Enable depth testing
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.main_window_name = "Visualizer"
        cv2.namedWindow(self.main_window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Roll", self.main_window_name, 0, 180, lambda x: None)
        cv2.createTrackbar("Pitch", self.main_window_name, 0, 180, lambda x: None)
        cv2.createTrackbar("Yaw", self.main_window_name, 0, 180, lambda x: None)
        cv2.createTrackbar("X", self.main_window_name, 0, 1000, lambda x: None)
        cv2.createTrackbar("Y", self.main_window_name, 0, 1000, lambda x: None)
        cv2.createTrackbar("Z", self.main_window_name, 0, 1000, lambda x: None)
        cv2.setTrackbarMin("X", self.main_window_name, -1000)
        cv2.setTrackbarMin("Y", self.main_window_name, -1000)
        cv2.setTrackbarMin("Z", self.main_window_name, -1000)
        cv2.setTrackbarMin("Roll", self.main_window_name, -180)
        cv2.setTrackbarMin("Pitch", self.main_window_name, -180)
        cv2.setTrackbarMin("Yaw", self.main_window_name, -180)
        # Create other named windows
        cv2.namedWindow("Image Overlayed", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Aligned Frames", cv2.WINDOW_NORMAL)


    def visualize(self, color_image, depth_image, fx, fy, cx, cy):
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(np.asanyarray(depth_image), alpha=0.03),
            cv2.COLORMAP_INFERNO,
        )
        images_aligned = np.hstack((color_image, depth_colormap))
        image_overlayed = cv2.addWeighted(color_image, 0.5, depth_colormap, 0.5, 0)
        cv2.imshow("Image Overlayed", image_overlayed)
        cv2.imshow("Aligned Frames", images_aligned)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        viewmat = np.eye(4, dtype=np.float32)
        R = scipy_rot.from_euler(
            "xyz",
            [
                cv2.getTrackbarPos("Roll", self.main_window_name),
                cv2.getTrackbarPos("Pitch", self.main_window_name),
                cv2.getTrackbarPos("Yaw", self.main_window_name),
            ],
            degrees=True,
        ).as_matrix()
        t = np.array(
            [
                cv2.getTrackbarPos("X", self.main_window_name),
                cv2.getTrackbarPos("Y", self.main_window_name),
                cv2.getTrackbarPos("Z", self.main_window_name),
            ],
            dtype=np.float32, 
        ) # in cm
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t / 100.0
        viewmat[3, 3] = 1.0
        vertices = rgbd_to_vertex_data(
            color_image.astype(float) / 255,
            np.asanyarray(depth_image).astype(float),
            fx,
            fy,
            cx,
            cy,
        )
        viewmat = opencv_to_opengl_projection(K, viewmat, 640, 480, 0.1, 1000.0)
        self.prog["projection"].write(np.ascontiguousarray(viewmat, dtype="f4"))
        vbo = self.ctx.buffer(vertices.astype("f4").tobytes())
        vao = self.ctx.simple_vertex_array(self.prog, vbo, "in_vert", "in_color")
        fbo = self.ctx.simple_framebuffer((640, 480))
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        vao.render(moderngl.POINTS)
        data = fbo.read()
        data = np.frombuffer(data, dtype="B")
        img = data.reshape((480, 640, 3))
        img = cv2.flip(img, 0)
        cv2.imshow(self.main_window_name, img)
        key = cv2.waitKey(1) & 0xFF
        return key