from transformers import pipeline
import gradio as gr

model = pipeline("summarization", model="facebook/bart-large-cnn")

def predict(prompt):
    summary = model(prompt, max_length=130, min_length=30)[0]["summary_text"]
    return summary

with gr.Blocks() as demo:
    textbox = gr.Textbox(placeholder="Enter text block to summarize", lines=4)
    gr.Interface(fn=predict, inputs=textbox, outputs="text")

demo.launch()
