import subprocess
import gradio as gr
import torch
import numpy as np
from modules.utils import realsense_rgb_depth, halt

def gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("<h1 style='text-align: center;'>🐼 Franka Robot Grasping Machine 🐼</h1>")

        with gr.Row():
            target = gr.Textbox(
                        label="What to Grasp?"
                    )
        
        with gr.Row():
            button_fetch = gr.Button("Start Grasping!")
            button_halt = gr.Button("Freeze!")

        with gr.Row():
            rgb_image = gr.Image(
                                    label="Panda sees...",
                                    type="numpy",
                                    streaming=True) 

            depth_image = gr.Image(
                                    label="Panda measures...",
                                    type="numpy",
                                    streaming=True)

        with gr.Row():
            output_console = gr.Textbox(label="Panda says...", lines=1)

        button_fetch.click(realsense_rgb_depth, inputs=[target], outputs=[rgb_image,depth_image,output_console])
        button_halt.click(halt, inputs=[], outputs=[rgb_image,depth_image,output_console])



    return demo


if __name__ == "__main__":
    demo = gradio_ui()
    demo.launch()