from time import sleep, strftime, gmtime
from shutil import copy
import gradio as gr # type: ignore
import requests, os

# Hack to allow downloading files from localhost
gr.processing_utils.PUBLIC_HOSTNAME_WHITELIST.append("127.0.0.1")

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
def context_to_server(cur_topic, files):
    global context_changed

    if files is None:
        return [gr.Files(), gr.Files()]

    for file in files:
        cur_file = {'file': open(file, "rb")}
        resp = requests.post(f"http://127.0.0.1:5000/upload/{cur_topic}", files=cur_file)
        upload_resp = resp.json()
        if not upload_resp['status'] == "ok":
            file = upload_resp['file']
            raise gr.Error(f"Could not upload file {file}", duration=None)
    
    context_changed = True
    return [gr.Files(value=None), gr.Files(value=show_context_files(cur_topic))]


# Show the currently uploaded context files
def show_context_files(cur_topic):
    resp = requests.post(f"http://127.0.0.1:5000/context/{cur_topic}")
    files = resp.json()['files']

    output = []
    for file in files:
        output.append(f"http://127.0.0.1:5000/download/{cur_topic}/{file}")

    return output


# Download all context files from server
def dl_all_server_context(cur_topic, files_list):
    if files_list is None:
        return

    if not os.path.exists("./saved_context"):
        try:
            os.mkdir("./saved_context")
            os.mkdir(f"./saved_context/{cur_topic}")
        except:
            pass
    elif not os.path.exists(f"./saved_context/{cur_topic}"):
        try:
            os.mkdir(f"./saved_context/{cur_topic}")
        except:
            pass

    for file in files_list:
        fname = os.path.basename(file)
        new_path = f"./saved_context/{cur_topic}/{fname}"
        if not os.path.exists(new_path):
            copy(file, new_path)


# Delete single context file
def delete_single_context(cur_topic, deleted: gr.DeletedFileData):
    global context_changed

    fname = os.path.basename(deleted.file.path)
    resp = requests.post(f"http://127.0.0.1:5000/delete/{cur_topic}/{fname}")
    del_resp = resp.json()
    if del_resp['status'] == "error":
        raise gr.Error(f"Error deleting file {fname}", duration=None)
    
    context_changed = True


# Delete all server context files
def delete_all_context(cur_topic, files_list):
    global context_changed

    if files_list is None:
        return gr.Files(value=None)

    for file in files_list:
        fname = os.path.basename(file)
        resp = requests.post(f"http://127.0.0.1:5000/delete/{cur_topic}/{fname}")
        del_resp = resp.json()
        if del_resp['status'] == "error":
            raise gr.Error(f"Error deleting file {fname}", duration=None)

    context_changed = True
    return gr.Files(value=None)


# Save history to local file
def history_to_local(cur_topic, history, fname):
    if not fname:
        cur_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        fname = f"ChatHistory_{cur_time}"
    
    if not os.path.exists("./histories"):
        try:
            os.mkdir("./histories")
            os.mkdir(f"./histories/{cur_topic}")
        except:
            pass
    elif not os.path.exists(f"./histories/{cur_topic}"):
        os.mkdir(f"./histories/{cur_topic}")

    with open(f"./histories/{cur_topic}/{fname}.txt", "w", encoding="utf-8") as outf:
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


# Ensure customization folder exists
def ensure_customization():
    if not os.path.exists("./customization"):
        try:
            os.mkdir("./customization")
        except:
            pass


# Update theme color
def update_theme_color(cur_color):
    ensure_customization()

    with open("./customization/theme_color.txt", "w", encoding="utf-8") as outf:
        outf.write(cur_color)


# Javascript needed for changing theme mode
def theme_mode_js():
    return """
    () => {
        document.body.classList.toggle('dark');
    }
    """


# Update chat layout
def update_chat_layout(cur_layout):
    ensure_customization()

    with open("./customization/chat_layout.txt", "w", encoding="utf-8") as outf:
        outf.write(cur_layout)
    
    return gr.Chatbot(type="messages", layout=cur_layout)


# Get all topics
def get_all_topics():
    ensure_customization()

    try:
        with open("./customization/all_topics.txt") as inf:
            all_topics = []
            for line in inf:
                all_topics.append(line.strip())
    except:
        all_topics = ["Default"]
    
    return all_topics


# Get current topic
def get_current_topic():
    ensure_customization()

    current_topic = "Default"
    try:
        with open("./customization/current_topic.txt") as inf:
            current_topic = inf.readline().strip()
    except:
        pass

    return current_topic


# Save current topic
def save_current_topic(cur_topic):
    ensure_customization()

    try:
        with open("./customization/current_topic.txt", "w", encoding="utf-8") as outf:
            outf.write(f"{cur_topic}\n")
    except:
        pass


# Detele currently selected topic, other than Default
def delete_current_topic(cur_topic, embedding_choice):
    if cur_topic == "Default":
        gr.Warning("Cannot delete default topic.", duration=None)
        return gr.Dropdown()

    data = {"embedding_choice": embedding_choice}
    resp = requests.post(f"http://127.0.0.1:5000/delete_topic/{cur_topic}", data=data)
    del_resp = resp.json()
    if del_resp["status"] == "error":
        raise gr.Error(del_resp["issue"], duration=None)

    topics = []
    try:
        with open("./customization/all_topics.txt", "r") as inf:
            for line in inf:
                topics.append(line.strip())
        topics.remove(cur_topic)
    except:
        pass

    try:
        with open("./customization/all_topics.txt", "w", encoding="utf-8") as outf:
            for topic in topics:
                outf.write(f"{topic}\n")
    except:
        pass

    save_current_topic(cur_topic)

    return gr.Dropdown(topics, value="Default")


# Make new topic
def make_new_topic(new_topic, embedding_choice):
    data = {"embedding_choice": embedding_choice}
    resp = requests.post(f"http://127.0.0.1:5000/new_topic/{new_topic}", data=data)
    mk_resp = resp.json()
    if not mk_resp["status"] == "ok":
        raise gr.Error("Issue making new topic", duration=None)
    
    topics = []
    if not os.path.exists("./customization/all_topics.txt"):
        topics.append("Default")
    else:
        try:
            with open("./customization/all_topics.txt", "r") as inf:
                for line in inf:
                    topics.append(line.strip())
        except:
            pass
    topics.append(new_topic)

    try:
        with open("./customization/all_topics.txt", "w", encoding="utf-8") as outf:
            for topic in topics:
                outf.write(f"{topic}\n")
    except:
        pass

    save_current_topic(new_topic)

    return [gr.Textbox(value=""), gr.Dropdown(topics, value=new_topic)]


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
def run_rag(message, history, cur_topic, use_history, embedding_choice, model_choice, num_docs, chunk_size, chunk_overlap, refresh_db, uploaded_history):
    """Ensure the RAG system uses the desired setup, then request an answer from the system.

    Args:
      message: Current query to send to the RAG system.
      history: OpenAI style of conversation history for this session.
      cur_topic: Which topic directory to use
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
    resp = requests.post(f"http://127.0.0.1:5000/setup/{cur_topic}", data=setup_info)
    setup_response = resp.json()
    if setup_response["status"] == "error":
        return setup_response['issue']

    if use_history and history:
        # Summarize history
        hist_info = {"user_history": history}
        resp = requests.post(f"http://127.0.0.1:5000/response/{cur_topic}", data=hist_info)
        hist_summary = resp.json()['response']

        # Send query and history summary
        resp_info = {"user_query": message, "user_history": hist_summary}
        resp = requests.post(f"http://127.0.0.1:5000/response/{cur_topic}", data=resp_info)
        response = resp.json()['response']
    else:
        # Send query to RAG system
        resp_info = {"user_query": message}
        resp = requests.post(f"http://127.0.0.1:5000/response/{cur_topic}", data=resp_info)
        response = resp.json()['response']

    return response


# Get customization options, to allow storing/reloading UI theme
def get_setup_vars():
    """Create variables to pass to UI Block object.

    Returns:
      CSS, color option, theme object, and chat layout to give to UI Block object.
    """
    # Add custom css
    css = """
    footer {visibility: hidden}
    """

    # Get currently set theme color
    saved_color = "rose"
    try:
        ensure_customization()
        with open("./customization/theme_color.txt", "r") as inf:
            saved_color = inf.readline().strip()
    except:
        pass

    # Specify theme to use
    theme =  gr.themes.Default(
        primary_hue=saved_color,
        secondary_hue=saved_color
    ).set(
        color_accent_soft='*primary_700',
        color_accent_soft_dark='*primary_700',
        border_color_primary='*primary_800',
        border_color_primary_dark='*primary_800',
        border_color_accent='*primary_950',
        border_color_accent_dark='*primary_950'
    )

    cur_layout = "bubble"
    try:
        with open("./customization/chat_layout.txt", "r") as inf:
            cur_layout = inf.readline().strip()
    except:
        pass

    return (css, saved_color, theme, cur_layout)


# Layout for the UI
def setup_layout(css, saved_color, theme, cur_layout):
    """Create Block object to be used for UI.

    Args:
      css: String with custom CSS.
      saved_color: String naming the saved color.
      theme: Customized Default theme.
      cur_layout: String naming which layout to use for the chatbot.
    
    Returns:
      The Block object to be used for the UI.
    """
    ensure_customization()
    with gr.Blocks(title="RAG System", theme=theme, css=css) as app:
        # Settings
        with gr.Row():
            with gr.Column(scale=1):
                # Current topic
                with gr.Column():
                    all_topics = get_all_topics()
                    current_topic = get_current_topic()
                    cur_topic = gr.Dropdown(
                        all_topics,
                        value=current_topic,
                        label="Current Topic"
                    )

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
                        chunk_size = gr.Number(
                            value=500,
                            label="Chunk Size",
                            info="Min value is 1.",
                            visible=False
                        )
                        chunk_overlap = gr.Number(
                            value=50,
                            label="Chunk Overlap",
                            info="Min value is 0.",
                            visible=False
                        )
                    reset_chunk_opts = gr.Button(
                        value="",
                        icon="./customization/reset.png",
                        variant="primary",
                        min_width=0,
                        visible=False
                    )

            # Main interface options
            with gr.Column(scale=4):
                # Tab containing chatbot and related options
                with gr.Tab(label="RAG Chat"):
                    # Chat history options
                    with gr.Row(equal_height=True):
                        history_file = gr.File(
                            show_label=False,
                            height=65
                        )
                        upload_history = gr.Button(
                            value="Upload History to Chat"
                        )
                        save_name = gr.Textbox(
                            placeholder="Enter desired name for file",
                            label="Chat History File Name"
                        )
                        save_history = gr.Button(
                            value="Save Chat History",
                            min_width=0
                        )
                        view_history = gr.Chatbot(
                            type="messages",
                            visible=False
                        )

                    # Chat interface
                    # Chatbox and Textbox specified to allow customization
                    main_chat = gr.ChatInterface(
                        run_rag,
                        type="messages",
                        chatbot=gr.Chatbot(
                            type="messages",
                            show_label=False,
                            height=550,
                            avatar_images=(None, None),
                            placeholder="# Welcome to the experimental RAG system!",
                            layout=cur_layout
                        ),
                        textbox=gr.Textbox(
                            type='text',
                            placeholder='Enter a query...',
                            show_label=False
                        ),
                        additional_inputs=[
                            cur_topic,
                            use_history,
                            embedding_choice,
                            model_choice,
                            num_docs,
                            chunk_size,
                            chunk_overlap,
                            refresh_db,
                            view_history
                        ]
                    )
                
                # Tab containing context file information
                with gr.Tab(label="Manage Context Files") as context_tab:
                    # Topic management
                    with gr.Row(equal_height=True):
                        delete_topic = gr.Button(
                            value="Delete Current Topic",
                            variant="stop"
                        )
                        new_topic_name = gr.Textbox(
                            label="New Topic Name",
                            placeholder="Enter name for new topic"
                        )
                        make_topic = gr.Button(
                            value="Make New Topic",
                            variant="primary"
                        )

                    # Context file management
                    with gr.Row():
                        # Upload context files
                        with gr.Column():
                            gr.Markdown(value="<center><h1>Upload Context Files</h1></center>")
                            upload_context = gr.Button(
                                value="Upload to RAG Server",
                                variant="primary"
                            )
                            context_files = gr.Files(
                                show_label=False
                            )
                        
                        # View current context files
                        with gr.Column():
                            gr.Markdown(value="<center><h1>Context Files on Server</h1></center>")
                            with gr.Row(equal_height=True):
                                download_all_context = gr.Button(
                                    value="Download All Context Files"
                                )
                                purge_context = gr.Button(
                                    value="Purge All Context Files",
                                    variant="stop"
                                )

                            view_context_files = gr.Files(
                                show_label=False
                            )

        # Customization options
        with gr.Sidebar(width=200, open=False, position="right"):
            # Settings that don't need a restart
            with gr.Column():
                chat_layout = gr.Radio(
                    ["panel", "bubble"],
                    value=cur_layout,
                    label="Chat Style"
                )
                theme_mode = gr.Button(
                    value="Toggle Dark Mode",
                    variant="primary"
                )

            # Settings that do need a restart            
            with gr.Group():
                gr.Markdown("Require restart:")
                theme_color = gr.Dropdown(
                    ["slate", "gray", "zinc", "neutral", "stone", "red", "orange", "amber", "yellow", "lime", "green", "emerald", "teal", "cyan", "sky", "blue", "indigo", "violet", "purple", "fuchsia", "pink", "rose"],
                    value=saved_color,
                    label="Theme Color",
                    interactive=True
                )
                reload_app = gr.Button(
                    value="Reload UI\n(Requires refreshing tab)",
                    variant="stop"
                )

        # Handle topics
        cur_topic.change(save_current_topic, inputs=[cur_topic])
        delete_topic.click(delete_current_topic, inputs=[cur_topic, embedding_choice], outputs=[cur_topic])
        make_topic.click(make_new_topic, inputs=[new_topic_name, embedding_choice], outputs=[new_topic_name, cur_topic])

        # Handle context files
        context_tab.select(show_context_files, inputs=[cur_topic], outputs=[view_context_files])
        upload_context.click(context_to_server, inputs=[cur_topic, context_files], outputs=[context_files, view_context_files])
        download_all_context.click(dl_all_server_context, inputs=[cur_topic, view_context_files])
        view_context_files.delete(delete_single_context, inputs=[cur_topic])
        purge_context.click(delete_all_context, inputs=[cur_topic, view_context_files], outputs=[view_context_files])

        # Handle chat history
        save_history.click(history_to_local, inputs=[cur_topic, main_chat.chatbot, save_name])
        upload_history.click(local_to_history, inputs=[history_file], outputs=[view_history, main_chat.chatbot])

        # Handle general options
        embedding_type.change(update_embedding_opts, inputs=[embedding_type], outputs=[embedding_choice])
        model_type.change(update_chat_opts, inputs=[model_type], outputs=model_choice)
        refresh_db.change(show_chunk_opts, inputs=[refresh_db], outputs=[chunk_size, chunk_overlap, reset_chunk_opts])
        reset_chunk_opts.click(chunk_opt_defaults, outputs=[chunk_size, chunk_overlap])

        # Handle customization options
        chat_layout.change(update_chat_layout, inputs=[chat_layout], outputs=[main_chat.chatbot])
        mode_js = theme_mode_js()
        theme_mode.click(None, js=mode_js)
        theme_color.change(update_theme_color, inputs=[theme_color])
        reload_app.click(update_reload)

    return app



if __name__ == "__main__":
    while True:
        # Launch UI with customization settings
        css, saved_color, theme, cur_layout = get_setup_vars()
        app = setup_layout(css, saved_color, theme, cur_layout)
        app.launch(
            favicon_path="./customization/favicon.png",
            share=False,
            server_name="0.0.0.0",
            prevent_thread_lock=True
        )

        # Loop until restart requested
        while not need_restart:
            sleep(0.5)

        # Handle restart
        need_restart = False
        app.close()