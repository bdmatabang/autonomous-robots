import pyrealsense2 as rs

# Initialize pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# Wait for the first frame to arrive
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()

# Get camera intrinsics
depth_intrin = depth_frame.profile.as_video_stream_profile().get_intrinsics()

# Print intrinsic parameters
print(f"Width: {depth_intrin.width}")
print(f"Height: {depth_intrin.height}")
print(f"Principal Point (cx, cy): ({depth_intrin.ppx}, {depth_intrin.ppy})")
print(f"Focal Length (fx, fy): ({depth_intrin.fx}, {depth_intrin.fy})")
print(f"Distortion Coefficients: {depth_intrin.coeffs}")