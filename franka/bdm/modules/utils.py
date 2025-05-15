
import os
import time
import numpy as np
import cv2
import gradio as gr
from ultralytics import YOLOWorld
from PIL import Image, ImageDraw
# import pyrealsense2 as rs

# Initializations
model = YOLOWorld("yolov8m-world.pt")
model.to('cuda')
go = 0
color_frame = None
"""
def realsense_rgb_depth(model, pipeline, config, target):
    global go
    go == 1

    # Enable both color and depth streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start the stream
    pipeline.start(config)


    while go == 1:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        model.set_classes([target])

        # Convert to NumPy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())  # uint16, depth in mm
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        results = model.predict(color_image)
        img_with_overlay = results[0].cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        rgbdraw = ImageDraw.Draw(color_image)
        depthdraw = ImageDraw.Draw(depth_image)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers (if they are not already)
            # Draw a rectangle (bounding box) on the image
            rgbdraw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            depthdraw.rectangle([x1, y1, x2, y2], outline="white", width=3)

        yield gr.update(value=color_image) , gr.update(value=depth_image)
        """

def webcam_rgb_depth(target):
    global go
    global model
    global color_frame
    if target == "":
        print("Please enter a target to hunt!")
    
    if target != "":
        go = 1
        model.set_classes([target])
        print("Target set to: ", target)
        while go == 1:
            # Capture frames from the laptop's webcam
            cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
            ret, color_frame = cap.read()
            # Convert the color frame to RGB format
            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2GRAY)

            if not ret:
                print("Can't detect camera")

            results = model.predict(color_frame)
            annotated_img = results[0].plot()

            yield gr.update(value=annotated_img) , gr.update(value=gray_frame)

def halt():
    global go
    global color_frame
    go = 0
    while go == 0:
        if color_frame:
            gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2GRAY)
            yield gr.update(value=color_frame) , gr.update(value=gray_frame)
        if color_frame is None:
            print("Can't freeze, have you tried turning it off and on again?")
            break