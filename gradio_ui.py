import gradio as gr
from gradio_webrtc import WebRTC
from functions import process_frame,process_uploaded_video

css = """.my-group {max-width: 300px !important; max-height: 300px !important;}
         .my-column {display: flex !important; justify-content: center !important; align-items: center !important;}"""

# Main interface
with gr.Blocks() as interface:
    gr.Markdown("# Emotion Detection")
    with gr.Tabs():
        with gr.Tab("Live Video"):
            gr.Markdown("### Live Video Interface")
            image = WebRTC(label="Stream", mode="send-receive", modality="video",width=600,height=500)
            image.stream(
                fn=process_frame,
                inputs=[image],
                outputs=[image]
            )

        with gr.Tab("Upload Video"):
            gr.Markdown("### Upload Video Interface")
            video_upload = gr.Video(label="Upload a Video",width=600,height=500)
            predict_btn = gr.Button("Predict")
            output_video = gr.Video(label="Processed Video",width=600,height=500,show_download_button=True)
            predict_btn.click(process_uploaded_video, inputs=video_upload, outputs=output_video)
