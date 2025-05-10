import pyrealsense2 as rs
import numpy as np

class RelsenseCamera:
    def __init__(self):
        self.rs = rs
        self.reset()
        self.config = rs.config()
        self.pipeline = rs.pipeline()
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image

    def stop(self):
        self.pipeline.stop()

    def reset(self):
        rs_ctx = self.rs.context()
        devices = rs_ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()

    def get_intrinsics(self):
        color_frame = self.pipeline.get_active_profile().get_stream(rs.stream.color)
        intr = color_frame.as_video_stream_profile().intrinsics
        fx = intr.fx
        fy = intr.fy
        cx = intr.ppx
        cy = intr.ppy
        return fx, fy, cx, cy