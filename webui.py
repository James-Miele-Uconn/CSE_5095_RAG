import gradio as gr # type: ignore
import requests

css = """
footer {visibility: hidden}
"""

# Specify theme to use
theme =  gr.themes.Default(
    primary_hue="rose",
    secondary_hue="rose"
).set(
    color_accent_soft_dark='*primary_800',
    border_color_primary_dark='*primary_800',
    border_color_accent_dark='*primary_800'
)

# Options lists for each type of embedding model
embeddings_dict = {
    "ollama": ["mxbai-embed-large", "nomic-embed-text"],
    "local": ["nomic-embed-text-v1.5", "bert-base-uncased"],
    "online": ["openai"]
}

# Options lists for each type of chat model
models_dict = {
    "ollama": ["deepseek-r1:7b", "deepseek-r1:14b", "deepseek-r1:32b", "deepseek-r1:70b", "llama3.3", "mistral", "mixtral:8x7b", "deepseek-r1:671b"],
    "local": ["bert-base-uncased", "gpt2", "Mistral-7B-Instruct-v0.3", "zephyr-7b-beta", "DarkForest-20B-v3.0"],
    "online": ["openai"]
}

# Function to update embedding options based on currently chosen type
def update_embedding_opts(embedding_type):
    new_opts = embeddings_dict[embedding_type.lower()]
    return gr.Dropdown(new_opts, value=new_opts[0])

# Function to update model options based on currently chosen type
def update_chat_opts(model_type):
    new_opts = models_dict[model_type.lower()]
    return gr.Dropdown(new_opts, value=new_opts[0])

def show_chunk_opts(rebuild_val):
    new_size = gr.Number(visible=rebuild_val)
    new_overlap = gr.Number(visible=rebuild_val)
    new_reset = gr.Button(visible=rebuild_val)

    return [new_size, new_overlap, new_reset]
    

def chunk_opt_defaults():
     return [gr.Number(value=500), gr.Number(value=50)]

def run_rag(message, history, use_history, embedding_choice, model_choice, num_docs, chunk_size, chunk_overlap, refresh_db):
    """Ensure the RAG system uses the desired setup, then request an answer from the system.

    Args:
      message: Current query to send to the RAG system.
      history: OpenAI style of conversation history for this session.
      use_history: Whether to use summary of chat history in query response.
      embedding_choice: Currently chosen embedding model to use.
      model_choice: Currently chosen chat model to use.
      num_docs: Number of chunks to use when creating an answer.
      chunk_size: Size of chunks to use for database chunks.
      chunk_overlap: Amount of overlap to use for database chunks.
      refresh_db: Whether the database should be forcibly refreshed.
    
    Returns:
      Formatted string response to the given query.
    """
    # Send info for RAG system setup
    setup_info = {
        "embedding_choice": embedding_choice,
        "model_choice": model_choice,
        "num_docs": num_docs,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "refresh_db": refresh_db
    }
    resp = requests.post("http://127.0.0.1:5000/setup", data=setup_info)
    setup_status = resp.json()['status']
    if setup_status != "ok":
        return "RAG system setup failed."

    if use_history and history:
        # Summarize history
        hist_info = {"user_history": history}
        resp = requests.post('http://127.0.0.1:5000/response', data=hist_info)
        hist_summary = resp.json()['response']

        # Send query and history summary
        resp_info = {"user_query": message, "user_history": hist_summary}
        resp = requests.post('http://127.0.0.1:5000/response', data=resp_info)
        response = resp.json()['response']
    else:
        # Send query to RAG system
        resp_info = {"user_query": message}
        resp = requests.post('http://127.0.0.1:5000/response', data=resp_info)
        response = resp.json()['response']

    return response

# Layout for the UI
with gr.Blocks(title="RAG System", theme=theme, css=css) as app:
    with gr.Row():
        # Settings
        with gr.Sidebar(width=500):
            # Embedding model options
            with gr.Column():
                    embedding_type = gr.Radio(
                        ["Ollama", "Local", "Online"],
                        value="Ollama",
                        label="Embedding Model Type"
                    )
                    embedding_choice = gr.Dropdown(
                        embeddings_dict["ollama"],
                        value=embeddings_dict["ollama"][0],
                        label="Embedding Model"
                    )

            # Chat model options
            with gr.Column():
                    model_type = gr.Radio(
                        ["Ollama", "Local", "Online"],
                        value="Ollama",
                        label="Chat Model Type"
                    )
                    model_choice = gr.Dropdown(
                        models_dict["ollama"],
                        value=models_dict["ollama"][0],
                        label="Chat Model"
                    )

            # Various other common options
            with gr.Column():
                num_docs = gr.Slider(
                    1,
                    10,
                    value=5,
                    step=1,
                    label="Number of chunks to use when answering query"
                )
                use_history = gr.Checkbox(
                     label="Add summary of chat history to query",
                     value=False
                )

            # Database options
            with gr.Group():
                refresh_db = gr.Checkbox(
                    value=False,
                    label="Force Rebuild Database"
                )

                with gr.Row():
                    with gr.Row():
                        chunk_size = gr.Number(
                            value=500,
                            label="Chunk Size",
                            info="Min value is 1.",
                            scale=6,
                            visible=False
                        )
                        chunk_overlap = gr.Number(
                            value=50,
                            label="Chunk Overlap",
                            info="Min value is 0.",
                            scale=6,
                            visible=False
                        )
                        reset_chunk_opts = gr.Button(
                            value="",
                            icon="./imgs/reset.png",
                            scale=1,
                            min_width=0,
                            visible=False
                        )

        # Chat interface
        # Chatbox and Textbox specified to allow customization
        with gr.Column(scale=2):
            main_chat = gr.ChatInterface(
                run_rag,
                type="messages",
                chatbot=gr.Chatbot(
                    type="messages",
                    show_label=False,
                    height=800,
                    avatar_images=(None, None),
                    value=[{"role": "assistant", "content": "Welcome to the experimental RAG system! How can I help?"}]
                ),
                textbox=gr.Textbox(
                    type='text',
                    placeholder='Enter a query...',
                    show_label=False
                ),
                additional_inputs=[
                    use_history,
                    embedding_choice,
                    model_choice,
                    num_docs,
                    chunk_size,
                    chunk_overlap,
                    refresh_db
                ],
            )

    # Allow updating the embedding/chat model options based on the relevant type currently selected
    embedding_type.change(update_embedding_opts, inputs=[embedding_type], outputs=[embedding_choice])
    model_type.change(update_chat_opts, inputs=[model_type], outputs=model_choice)

    # Allow showing/hiding database options based on checkbox
    refresh_db.change(show_chunk_opts, inputs=[refresh_db], outputs=[chunk_size, chunk_overlap, reset_chunk_opts])

    # Allow resetting chunk options to default values
    reset_chunk_opts.click(chunk_opt_defaults, outputs=[chunk_size, chunk_overlap])


if __name__ == "__main__":
    # Allow use on local network, may add flag to either run locally or on network
    app.launch(
        favicon_path='./imgs/favicon.png',
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )