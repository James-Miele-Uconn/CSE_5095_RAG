from time import sleep, strftime, gmtime
import gradio as gr # type: ignore
import requests

# Global values for state tracking
need_restart = False
just_uploaded = False

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


# Save history to local file
def history_to_local(history):
    cur_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    fname = f"ChatHistory_{cur_time}"
    with open(f"./output_files/{fname}.txt", "w", encoding="utf-8") as outf:
        outf.write(str(history))


# Load local file to history
def local_to_history(history_file):
    global just_uploaded

    history = []
    try:
        with open(history_file, "r") as inf:
            history = inf.readlines()
            if history:
                history = eval(history[0])
        just_uploaded = True
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


# Update embedding options based on currently chosen type
def update_embedding_opts(embedding_type):
    new_opts = embeddings_dict[embedding_type.lower()]
    return gr.Dropdown(new_opts, value=new_opts[0])


# Update model options based on currently chosen type
def update_chat_opts(model_type):
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
    global just_uploaded
    if just_uploaded:
        for idx in range(len(uploaded_history)):
            history.insert(idx, uploaded_history[idx])
        just_uploaded = False

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


# Get customization options, to allow storing/reloading UI theme
def get_setup_vars():
    """Create variables to pass to UI Block object.

    Returns:
      CSS, color option, and theme object to give to UI Block object.
    """
    # Add custom css
    css = """
    footer {visibility: hidden}
    """

    # Get currently set theme color
    saved_color = "rose"
    try:
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

    return (css, saved_color, theme)


# Layout for the UI
def setup_layout(css, saved_color, theme):
    """Create Block object to be used for UI.

    Args:
      css: String with custom CSS.
      saved_color: String naming the saved color.
      theme: Customized Default theme.
    
    Returns:
      The Block object to be used for the UI.
    """
    with gr.Blocks(title="RAG System", theme=theme, css=css) as app:
        with gr.Row():
            # Settings
            with gr.Sidebar(width=500, open=False):
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
                                icon="./customization/reset.png",
                                variant="primary",
                                scale=1,
                                min_width=0,
                                visible=False
                            )

            # Main interface options
            with gr.Column(scale=2):
                # Tab containing chatbot and related options
                with gr.Tab(label="RAG Chat"):
                    # Chat history options
                    with gr.Row():
                        history_file = gr.File(
                            label="Chat History File",
                            height=65
                        )
                        upload_history = gr.Button(
                            value="Upload History to Chat"
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
                            placeholder="# Welcome to the experimental RAG system!"
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
                            refresh_db,
                            view_history
                        ]
                    )

        # Customization options
        with gr.Sidebar(width=200, open=False, position="right"):
            theme_color = gr.Dropdown(
                ["slate", "gray", "zinc", "neutral", "stone", "red", "orange", "amber", "yellow", "lime", "green", "emerald", "teal", "cyan", "sky", "blue", "indigo", "violet", "purple", "fuchsia", "pink", "rose"],
                value=saved_color,
                label="Theme Color",
                info="Change requires restart",
                interactive=True
            )
            reload_app = gr.Button(
                value="Reload UI\n(Requires refreshing tab)",
                variant="stop"
            )

        # Save chat history to file
        save_history.click(history_to_local, inputs=[main_chat.chatbot])

        # Upload chat history file to chatbot
        upload_history.click(local_to_history, inputs=[history_file], outputs=[view_history, main_chat.chatbot])

        # Update the embedding/chat models options based on the relevant types currently selected
        embedding_type.change(update_embedding_opts, inputs=[embedding_type], outputs=[embedding_choice])
        model_type.change(update_chat_opts, inputs=[model_type], outputs=model_choice)

        # Show/hide database options based on checkbox
        refresh_db.change(show_chunk_opts, inputs=[refresh_db], outputs=[chunk_size, chunk_overlap, reset_chunk_opts])

        # Reset chunk options to default values
        reset_chunk_opts.click(chunk_opt_defaults, outputs=[chunk_size, chunk_overlap])

        # Save color on theme color change
        theme_color.change(update_theme_color, inputs=[theme_color])

        # Restart UI, still requires browser tab to be refreshed
        reload_app.click(update_reload)

    return app



if __name__ == "__main__":
    while True:
        # Launch UI with customization settings
        css, saved_color, theme = get_setup_vars()
        app = setup_layout(css, saved_color, theme)
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