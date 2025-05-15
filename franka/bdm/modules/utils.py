
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
#pipeline = rs.pipeline()
#config = rs.config()
go = 0
color_frame = None
"""
def realsense_rgb_depth(target):
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
            # Enable both color and depth streams
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            # Start the stream
            pipeline.start(config)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
            color_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.convertScaleAbs(depth_image, alpha=0.03)
            depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)

            results = model.predict(color_frame)
            annotated_img = results[0].plot()

        yield gr.update(value=annotated_img) , gr.update(value=depth_colormap)
        """

def webcam_rgb_depth(target):
    global go
    global model
    global color_frame
    if target == "":
        yield gr.update(value=None) , gr.update(value=None), gr.update(value="Please enter a target to hunt!")
    
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
                yield gr.update(value=None) , gr.update(value=None), gr.update(value="Can't detect camera!")

            results = model.predict(color_frame)
            annotated_img = results[0].plot()

            yield gr.update(value=annotated_img) , gr.update(value=gray_frame), gr.update(value="I am hunting for " + target)

def halt():
    global go
    global color_frame
    go = 0
    while go == 0:
        if color_frame:
            gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2GRAY)
            yield gr.update(value=color_frame) , gr.update(value=gray_frame)
        if color_frame is None:
            yield gr.update(value=None) , gr.update(value=None), gr.update(value="Can't freeze, have you tried turning it off and on again?")
            break