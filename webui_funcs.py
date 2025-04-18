from time import strftime, gmtime
import gradio as gr # type: ignore
import requests, os


# Global values for state tracking
need_restart = False
history_uploaded = False
context_changed = False

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


# Load context files to rag server
def context_to_server(files):
    global context_changed

    if files is None:
        return [gr.Files(), gr.Files()]

    for file in files:
        cur_file = {'file': open(file, "rb")}
        resp = requests.post('http://127.0.0.1:5000/upload', files=cur_file)
        upload_resp = resp.json()
        if not upload_resp['status'] == "ok":
            file = upload_resp['file']
            raise gr.Error(f"Could not upload file {file}", duration=None)
    
    context_changed = True
    return [gr.Files(value=None), gr.Files(value=show_context_files())]


# Show the currently uploaded context files
def show_context_files():
    resp = requests.post('http://127.0.0.1:5000/context')
    files = resp.json()['files']

    output = []
    for file in files:
        output.append(f"http://127.0.0.1:5000/download/{file}")

    return output


# Save history to local file
def history_to_local(history, fname):
    if not fname:
        cur_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        fname = f"ChatHistory_{cur_time}"
    
    if not os.path.exists("./histories"):
        try:
            os.mkdir("./histories")
        except:
            pass

    with open(f"./histories/{fname}.txt", "w", encoding="utf-8") as outf:
        outf.write(str(history))


# Load local file to history
def local_to_history(history_file):
    global history_uploaded

    history = []
    try:
        with open(history_file, "r") as inf:
            history = inf.readlines()
            if history:
                history = eval(history[0])
        history_uploaded = True
    except:
        pass

    return [gr.Chatbot(type="messages", value=history), gr.Chatbot(type="messages", value=history)]


# Update global variable checking for reload
def update_reload():
    global need_restart
    need_restart = True


# Update theme color
def update_theme_color(cur_color):
    with open("./customization/theme_color.txt", "w", encoding="utf-8") as outf:
        outf.write(cur_color)


# Update chat layout
def update_chat_layout(cur_layout):
    with open("./customization/chat_layout.txt", "w", encoding="utf-8") as outf:
        outf.write(cur_layout)
    
    return gr.Chatbot(type="messages", layout=cur_layout)


# Update embedding options based on currently chosen type
def update_embedding_opts(embedding_type):
    global embeddings_dict
    new_opts = embeddings_dict[embedding_type.lower()]
    return gr.Dropdown(new_opts, value=new_opts[0])


# Update model options based on currently chosen type
def update_chat_opts(model_type):
    global models_dict
    new_opts = models_dict[model_type.lower()]
    return gr.Dropdown(new_opts, value=new_opts[0])


# Show or hide chunk options
def show_chunk_opts(rebuild_val):
    new_size = gr.Number(visible=rebuild_val)
    new_overlap = gr.Number(visible=rebuild_val)
    new_reset = gr.Button(visible=rebuild_val)

    return [new_size, new_overlap, new_reset]


# Reset chunk options to default values
def chunk_opt_defaults():
     return [gr.Number(value=500), gr.Number(value=50)]


# Main chat function
def run_rag(message, history, use_history, embedding_choice, model_choice, num_docs, chunk_size, chunk_overlap, refresh_db, uploaded_history):
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
      uploaded_history: Uploaded history to add 
    
    Returns:
      Formatted string response to the given query.
    """
    global history_uploaded, context_changed

    # Add uploaded history to current history, if needed
    if history_uploaded:
        for idx in range(len(uploaded_history)):
            history.insert(idx, uploaded_history[idx])
        history_uploaded = False

    # Force database refresh if context information has changed
    if context_changed:
        refresh_db = True
        context_changed = False

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
    setup_response = resp.json()
    if setup_response['status'] == "error":
        return setup_response['issue']

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