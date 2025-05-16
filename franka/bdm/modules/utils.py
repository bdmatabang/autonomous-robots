
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
cx, cy = 327.1177978515625, 240.3525848388672
fx, fy = 390.7403259277344, 390.7403259277344
pipeline_status = False
go = 0
color_frame, color_image, depth_colormap = None, None, None
object_x, object_y, object_z = None, None, None
cam_offset_z = -0.02 # gripper is currently 2cm below cam
cam_offset_x = -0.05 # gripper is currently 5cm behind cam
originx, originy, originz = 0.3, 0, 0.45
x, y, z = cam_offset_x, 0, cam_offset_z

def move_robot(object_center_x,object_center_y,object_z):
    global x, y, z, originx, originy, originz
    # in camera coordinate system, positive x is rightward, positive y is downward
    # in panda coordinate system, positive x if forward, positive y is leftward, positive z is upward
    # movement in x will be pandax = -camy
    # movement in y will be panday = -camx 
    movex = -object_center_y
    movey = -object_center_x
    # python rest.py <x> <y> <z> <time> <delta_theta_z_deg> <delta_theta_y_deg> <delta_theta_x_deg>
    remotecommand = f'ssh researcher@192.168.1.129'
    changedirpy = f'cd ~/autonomous-robots/franka/python'
    remotecommand = f'python rest.py --parameters {movex} {movey} 0.45 7'
    rth = f'python rest.py --parameters {originx} {originy} {originz} 5'
    subprocess.run([remotecommand, '&&', changedirpy, '&&', remotecommand], shell=True)
    time.sleep(3)
    subprocess.run([rth], shell=True)
    subprocess.run('exit')


def realsense_rgb_depth(target):
    global go
    global model
    global color_image
    global pipeline, config, pipeline_status
    global cx, cy, fx, fy
    global object_x, object_y, object_z 
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if not pipeline_status:
        profile = pipeline.start(config)
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
                align_to = rs.stream.color
                align = rs.align(align_to)

                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
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


                depth_sensor = profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                depth_roi = np.asanyarray(depth_frame.get_data())[y_min:y_max, x_min:x_max] * depth_scale
                valid_depths = depth_roi[depth_roi > 0]


                cls_id = int(results[0].boxes[0].cls.cpu().numpy()[0])
                conf = float(results[0].boxes[0].conf.cpu().numpy()[0])
                cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label = f"{target} {conf:.2f}"
                cv2.putText(annotated_img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                center_x = int((x_min + x_max) / 2)
                center_y = int((y_min + y_max) / 2)
                half = 10 // 2
                x1 = max(center_x - half, 0)
                x2 = min(center_x + half + 1, depth_image.shape[1])
                y1 = max(center_y - half, 0)
                y2 = min(center_y + half + 1, depth_image.shape[0])
                region = depth_image[y1:y2, x1:x2]
                valid_region = region[region > 0]
                if len(valid_region) > 0:
                    object_z = valid_region.mean() * depth_scale
                else:
                    continue

                object_center_x = (center_x - cx) * object_z / fx
                object_center_y = (center_y - cy) * object_z / fx

                print(object_center_x,object_center_y,object_z)

                yield gr.update(value=annotated_img) , gr.update(value=depth_colormap), gr.update(value="I am hunting for " + target)
                move_robot(object_center_x,object_center_y,object_z)
                time.sleep(5)
            else:
                # If nothing was detected, just keep going
                yield gr.update(value=color_image), gr.update(value=depth_colormap), gr.update(value="No target detected!")


"""
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
            """

def halt():
    global go
    global color_image, depth_colormap
    global pipeline_status
    global object_x, object_y, object_z
    go = 0
    if pipeline_status:
        pipeline.stop()
        pipeline_status = False
    while go == 0:
        if color_image:
            yield gr.update(value=color_image) , gr.update(value=depth_colormap), gr.update(value="Stopped Streaming!")
        if color_image is None:
            yield gr.update(value=None) , gr.update(value=None), gr.update(value="Can't freeze, have you tried turning it off and on again?")
            break
"""
def grasp():
    global go
    global color_image, depth_colormap
    global pipeline_status
    global object_x, object_y, object_z
    go = 0
    if object_x:
"""
