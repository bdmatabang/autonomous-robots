
import os
import time
import numpy as np
import cv2
import gradio as gr
import subprocess
from ultralytics import YOLOWorld
from PIL import Image, ImageDraw
import pyrealsense2 as rs

# Initializations
model = YOLOWorld("yolov8m-world.pt")
model.to('cuda')
pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
pipeline_status = False
go = 0
color_frame = None
cam_offset = 0
x, y, z = 0, cam_offset, 0

def move_robot(objectctrxy, objectctrz):
    global x, y, z
    # in panda coordinate system, positive x if forward, positive y is leftward, positive z is upward
    # in camera coordinate system, positive x is rightward, positive y is downward
    # movement in x will be objectctrxy[1] - 
    #movex = 
    # python rest.py <x> <y> <z> <time> <delta_theta_z_deg> <delta_theta_y_deg> <delta_theta_x_deg>
    #command = 'python rest.py '
    #subprocess



def realsense_rgb_depth(target):
    global go
    global model
    global color_frame
    global pipeline, config, pipeline_status
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if not pipeline_status:
        pipeline.start(config)
        pipeline_status = True
    if target == "":
        pipeline.stop()
        pipeline_status = False
        yield gr.update(value=None) , gr.update(value=None), gr.update(value="Please enter a target to hunt!")
    
    if target != "":
        go = 1
        model.set_classes([target])
        while go == 1:
            try:
        
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not depth_frame or not color_frame:
                    print("CAN'T SEE BRO")
                
                color_image = np.asanyarray(color_frame.get_data())
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)


            except Exception as e:
                print(e)
                yield gr.update(value=None) , gr.update(value=None), gr.update(value="I can't see! Check the camera connection!")
                break

            results = model.predict(color_image)
            if results[0].boxes and len(results[0].boxes) > 0:
                annotated_img = color_image.copy()

                # Extract the first bounding box
                box = results[0].boxes[0].xyxy.cpu().numpy().astype(int)[0]  # [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = box
                cls_id = int(results[0].boxes[0].cls.cpu().numpy()[0])
                conf = float(results[0].boxes[0].conf.cpu().numpy()[0])
                cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label = f"{target} {conf:.2f}"
                cv2.putText(annotated_img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                center_x = int((x_min + x_max) / 2)
                center_y = int((y_min + y_max) / 2)
                object_ctr_xy = (center_x, center_y)

                # Get the depth value at the object's center
                object_ctr_z = depth_frame.get_distance(center_x, center_y)

                yield gr.update(value=annotated_img) , gr.update(value=depth_colormap), gr.update(value="I am hunting for " + target)
            else:
                # If nothing was detected, just keep going
                yield gr.update(value=color_image), gr.update(value=depth_colormap), gr.update(value="No target detected!")



def webcam_rgb_depth(target):
    global go
    global model
    global color_frame
    global pipeline, config, pipeline_status
    if target == "":
        yield gr.update(value=None) , gr.update(value=None), gr.update(value="Please enter a target to hunt!")
    
    if target != "":
        go = 1
        model.set_classes([target])
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
    global pipeline_status
    go = 0
    if pipeline_status:
        pipeline.stop()
        pipeline_status = False
    while go == 0:
        if color_frame:
            gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2GRAY)
            yield gr.update(value=color_frame) , gr.update(value=gray_frame)
        if color_frame is None:
            yield gr.update(value=None) , gr.update(value=None), gr.update(value="Can't freeze, have you tried turning it off and on again?")
            break