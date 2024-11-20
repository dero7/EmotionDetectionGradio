from fastapi import FastAPI
import gradio as gr
from gradio_ui import interface

app = FastAPI()

app = gr.mount_gradio_app(app,interface,path='/')