import gradio as gr
import time, requests

def echo(message, history, system_prompt, tokens):
    response = f"System prompt: {system_prompt}\n Message: {message}."
    for i in range(min(len(response), int(tokens))):
        time.sleep(0.05)
        yield response[: i + 1]


def my_func(message, history, system_prompt, tokens):
    query_info = {"query": message}
    resp = requests.post('http://127.0.0.1:5000', data=query_info)
    data = resp.json()
    response = data['response']
    return response

with gr.Blocks() as app:
    system_prompt = gr.Textbox("You are helpful AI.", label="System Prompt")
    slider = gr.Slider(10, 100, render=False)

    gr.ChatInterface(
        my_func,
        type="messages",
        additional_inputs=[system_prompt, slider],
    )

if __name__ == "__main__":
    app.launch()