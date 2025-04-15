import gradio as gr
import requests

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

def run_rag(message, history, embedding_choice, model_choice, num_docs, refresh_db):
    """Ensure the RAG system uses the desired setup, then request an answer from the system.

    Args:
      message: The current query to send to the RAG system.
      history: OpenAI style of conversation history for this session.
      embedding_choice: The currently chosen embedding model to use.
      model_choice: The currently chosen chat model to use.
      num_docs: The number of chunks to use when creating an answer.
      refresh_db: Whether the database should be forcibly refreshed.
    
    Returns:
      Formatted string response to the given query.
    """
    # Send info for RAG system setup
    setup_info = {
        "embedding_choice": embedding_choice,
        "model_choice": model_choice,
        "num_docs": num_docs,
        "refresh_db": refresh_db
    }
    resp = requests.post("http://127.0.0.1:5000/setup", data=setup_info)
    setup_status = resp.json()['status']
    if setup_status != "ok":
        return "RAG system setup failed."

    # Send query to RAG system
    query_info = {"query": message}
    resp = requests.post('http://127.0.0.1:5000', data=query_info)
    response = resp.json()['response']
    return response

# Specify theme to use
theme =  gr.themes.Default(
    primary_hue="rose",
    secondary_hue="rose"
).set(
    color_accent_soft_dark='*primary_800'
)

# Layout for the UI
with gr.Blocks(title="RAG System", theme=theme) as app:
    # General options, to be displayed above the chatbox
    with gr.Row():
        with gr.Column():
            embedding_type = gr.Dropdown(["Ollama", "Local", "Online"], value="Ollama", label="Embedding Model Type")
            embedding_choice = gr.Dropdown(embeddings_dict["ollama"], value=embeddings_dict["ollama"][0], label="Embedding Model")
        with gr.Column():
            model_type = gr.Dropdown(["Ollama", "Local", "Online"], value="Ollama", label="Chat Model Type")
            model_choice = gr.Dropdown(models_dict["ollama"], value=models_dict["ollama"][0], label="Chat Model")
        with gr.Column():
            num_docs = gr.Slider(1, 10, value=5, step=1, label="Number of chunks to use")
            refresh_db = gr.Checkbox(value=False, label="Force Rebuild Database")

    # Allow updating the embedding/chat model options based on the relevant type currently selected
    embedding_type.change(update_embedding_opts, inputs=[embedding_type], outputs=[embedding_choice])
    model_type.change(update_chat_opts, inputs=[model_type], outputs=model_choice)

    # Chat interface to use
    # Chatbox and Textbox specified to allow customization
    gr.ChatInterface(
        run_rag,
        type="messages",
        chatbot=gr.Chatbot(
            type="messages",
            show_label=False,
            avatar_images=(None, None),
            placeholder='Welcome to the Experimental RAG System!'
        ),
        textbox=gr.Textbox(
            type='text',
            placeholder='Enter a query...',
            show_label=False
        ),
        additional_inputs=[embedding_choice, model_choice, num_docs, refresh_db],
    )

if __name__ == "__main__":
    # Allow use on local network, may add flag to either run locally or on network
    app.launch(
        favicon_path='favicon.png',
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )