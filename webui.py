from time import sleep, strftime, gmtime
from shutil import copy
import gradio as gr # type: ignore
import pandas as pd
import requests, os, argparse


def parse_webui_arguments():
    """Parse command-line arguments for webui using argparse.

    Returns:
      All arguments that can be set through the cmd.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag_server_ip", metavar="", help="\twhat address to use for the rag server. Defaults to 127.0.0.1.", default="127.0.0.1", type=str)
    parser.add_argument("--rag_server_port", metavar="", help="\twhat port to use for the rag server. Defaults to 5000.", default="5000", type=str)
    args = parser.parse_args()

    return args

webui_args = parse_webui_arguments()
rag_ip = webui_args.rag_server_ip
rag_port = webui_args.rag_server_port

# Hack to allow downloading files from localhost
gr.processing_utils.PUBLIC_HOSTNAME_WHITELIST.append(rag_ip)
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost, 127.0.0.1/8, ::1"

# Global values for state tracking
need_restart = False
history_uploaded = False
context_changed = False


# Options lists for each type of embedding model
embeddings_dict = {
    "ollama": ["mxbai-embed-large", "snowflake-arctic-embed:335m", "nomic-embed-text"],
    "local": ["nomic-embed-text-v1.5", "bert-base-uncased"],
    "online": ["openai"]
}

# Options lists for each type of chat model
models_dict = {
    "ollama": ["deepseek-r1:7b", "deepseek-r1:14b", "deepseek-r1:32b", "deepseek-r1:70b", "r1-1776:70b", "llama3.3", "mistral:7b", "mistral:7b-instruct", "mistral:7b-instruct-fp16", "mixtral:8x7b"],
    "local": ["bert-base-uncased", "gpt2", "Mistral-7B-Instruct-v0.3", "zephyr-7b-beta", "DarkForest-20B-v2.0", "DarkForest-20B-v3.0"],
    "online": ["openai"]
}


# Load context files to rag server
def context_to_server(cur_topic, files):
    global context_changed

    if files is None:
        return [gr.Files(), gr.Files()]

    for file in files:
        cur_file = open(file, "rb")
        file_data = {'file': cur_file}
        resp = requests.post(f"http://{rag_ip}:{rag_port}/upload/{cur_topic}", files=file_data)
        upload_resp = resp.json()
        cur_file.close()
        if not upload_resp['status'] == "ok":
            file = upload_resp['file']
            raise gr.Error(f"Could not upload file {file}", duration=None)
    
    context_changed = True
    return [gr.Files(value=None), gr.Files(value=show_context_files(cur_topic))]


# Show the currently uploaded context files
def show_context_files(cur_topic):
    resp = requests.post(f"http://{rag_ip}:{rag_port}/context/{cur_topic}")
    files = resp.json()['files']

    output = []
    for file in files:
        output.append(f"http://{rag_ip}:{rag_port}/download/{cur_topic}/{file}")

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
            raise gr.Error("Could not make save directories.", duration=None)
    elif not os.path.exists(f"./saved_context/{cur_topic}"):
        try:
            os.mkdir(f"./saved_context/{cur_topic}")
        except:
            raise gr.Error("Could not make topic save directory.", duration=None)

    for file in files_list:
        fname = os.path.basename(file)
        new_path = f"./saved_context/{cur_topic}/{fname}"
        if not os.path.exists(new_path):
            copy(file, new_path)
    
    gr.Info("All context files downloaded")


# Delete single context file
def delete_single_context(cur_topic, deleted: gr.DeletedFileData):
    global context_changed

    fname = os.path.basename(deleted.file.path)
    resp = requests.post(f"http://{rag_ip}:{rag_port}/delete/{cur_topic}/{fname}")
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
        resp = requests.post(f"http://{rag_ip}:{rag_port}/delete/{cur_topic}/{fname}")
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

    try:
        with open(f"./histories/{cur_topic}/{fname}.txt", "w", encoding="utf-8") as outf:
            outf.write(str(history))
    except Exception as e:
        raise gr.Error(e, duration=None)
    
    gr.Info("History saved")


# Load local file to history
def local_to_history(history_file):
    global history_uploaded

    history = []
    try:
        with open(history_file, "r", encoding="utf-8") as inf:
            history = inf.readlines()
            if history:
                history = eval(history[0])
        history_uploaded = True
    except Exception as e:
        raise gr.Error(e, duration=None)

    output = [
        gr.Chatbot(type="messages", value=history),
        gr.Chatbot(type="messages", value=history),
        gr.File(value=None)
    ]

    return output


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
    with open("./customization/theme_color.txt", "w", encoding="utf-8") as outf:
        outf.write(cur_color)


# Update theme color
def update_avatar_size(cur_size):
    with open("./customization/avatar_size.txt", "w", encoding="utf-8") as outf:
        outf.write(cur_size)


# Update theme color
def update_avatar_shape(cur_shape):
    with open("./customization/avatar_shape.txt", "w", encoding="utf-8") as outf:
        outf.write(cur_shape)


# Update theme color
def update_avatar_fill(cur_fill):
    with open("./customization/avatar_fill.txt", "w", encoding="utf-8") as outf:
        outf.write(cur_fill)


# Add new user icons
def add_user_icons(cur_icons):
    for icon in cur_icons:
        fname = os.path.basename(icon)
        copy(icon, f"./customization/user_icons/{fname}")

    new_user_icon_opts = get_user_icon_opts()

    return [gr.Files(value=None), gr.Dropdown(new_user_icon_opts)]


# Remove all but default user icons
def purge_user_icons():
    for icon in os.listdir("./customization/user_icons"):
        fname = os.path.basename(icon)
        if fname != "default.png":
            try:
                os.remove(f"./customization/user_icons/{icon}")
            except Exception as e:
                raise gr.Error(e, duration=None)
    
    new_user_icon_opts = get_user_icon_opts()

    return gr.Dropdown(new_user_icon_opts, value=new_user_icon_opts[0])


# Add new chatbot icons
def add_chatbot_icons(cur_icons):
    for icon in cur_icons:
        fname = os.path.basename(icon)
        copy(icon, f"./customization/chatbot_icons/{fname}")

    new_chatbot_icon_opts = get_chatbot_icon_opts()

    return [gr.Files(value=None), gr.Dropdown(new_chatbot_icon_opts)]


# Remove all but default chatbot icons
def purge_chatbot_icons():
    for icon in os.listdir("./customization/chatbot_icons"):
        fname = os.path.basename(icon)
        if fname != "default.png":
            try:
                os.remove(f"./customization/chatbot_icons/{icon}")
            except Exception as e:
                raise gr.Error(e, duration=None)
    
    new_chatbot_icon_opts = get_chatbot_icon_opts()

    return gr.Dropdown(new_chatbot_icon_opts, value=new_chatbot_icon_opts[0])


# Javascript needed for changing theme mode
def theme_mode_js():
    return """
    () => {
        document.body.classList.toggle('dark');
    }
    """


# Update chat layout
def update_chat_layout(cur_layout):
    with open("./customization/chat_layout.txt", "w", encoding="utf-8") as outf:
        outf.write(cur_layout)
    
    return gr.Chatbot(type="messages", layout=cur_layout)


# Get all user icon options
def get_user_icon_opts():
    user_icons = []
    for file in os.listdir("./customization/user_icons"):
        user_icons.append(file)
    
    return user_icons


# Get saved user icon name
def get_saved_user_icon():
    saved_user_icon = "default.png"
    try:
        with open("./customization/user_icon.txt", "r", encoding="utf-8") as inf:
            saved_user_icon = inf.readline().strip()
    except:
        pass

    return saved_user_icon


# Get all chatbot icon options
def get_chatbot_icon_opts():
    chatbot_icons = []
    for file in os.listdir("./customization/chatbot_icons"):
        chatbot_icons.append(file)
    
    return chatbot_icons


# Get saved chatbot icon name
def get_saved_chatbot_icon():
    saved_chatbot_icon = "default.png"
    try:
        with open("./customization/chatbot_icon.txt", "r", encoding="utf-8") as inf:
            saved_chatbot_icon = inf.readline().strip()
    except:
        pass

    return saved_chatbot_icon


# Save user icon choice
def update_user_ai_icons(user_icon_file, chatbot_icon_file):
    with open("./customization/user_icon.txt", "w", encoding="utf-8") as outf:
        outf.write(user_icon_file)
    
    with open("./customization/chatbot_icon.txt", "w", encoding="utf-8") as outf:
        outf.write(chatbot_icon_file)
    
    user_icon_path = f"./customization/user_icons/{user_icon_file}"
    chatbot_icon_path = f"./customization/chatbot_icons/{chatbot_icon_file}"

    output = [
        gr.Chatbot(type="messages", avatar_images=(user_icon_path, chatbot_icon_path)),
        gr.Image(value=user_icon_path),
        gr.Image(value=chatbot_icon_path)
    ]

    return output


# Get all topics
def get_all_topics():
    ensure_customization()

    try:
        with open("./customization/all_topics.txt", "r") as inf:
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
    resp = requests.post(f"http://{rag_ip}:{rag_port}/delete_topic/{cur_topic}", data=data)
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
def make_new_topic(new_topic):
    resp = requests.post(f"http://{rag_ip}:{rag_port}/new_topic/{new_topic}")
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


# Show api keys
def show_api_keys(cur_topic):
    resp = requests.post(f"http://{rag_ip}:{rag_port}/get_api_keys/{cur_topic}")
    api_resp = resp.json()
    if api_resp["status"] == "error":
        raise gr.Error(api_resp["issue"], duration=None)
    
    if api_resp["data"] == "":
        df = pd.DataFrame()
    else:
        apis = eval(api_resp["data"])["0"]
        keys = eval(api_resp["data"])["1"]
        api_vals = [apis[f"{idx}"] for idx in sorted(apis.keys())]
        key_vals = [keys[f"{idx}"] for idx in sorted(keys.keys())]
        data = {"API": api_vals, "Key": key_vals}
        df = pd.DataFrame(data=data)

    return gr.Dataframe(value=df)


# Save api keys
def save_api_keys(cur_topic, api_keys):
    api_keys.to_csv("./temp_csv.csv", header=False, index=False)

    cur_file = open("./temp_csv.csv", "rb")
    file_data = {"file": cur_file}
    resp = requests.post(f"http://{rag_ip}:{rag_port}/save_api_keys/{cur_topic}", files=file_data)
    upload_resp = resp.json()
    if upload_resp["status"] == "error":
        raise gr.Error(upload_resp["issue"])
    
    try:
        cur_file.close()
        os.remove("./temp_csv.csv")
    except Exception as e:
        raise gr.Error(e.args[0], duration=None)
    gr.Info("API Keys Saved")


# Purge api keys
def purge_api_keys(cur_topic):
    resp = requests.post(f"http://{rag_ip}:{rag_port}/purge_api_keys/{cur_topic}")
    del_resp = resp.json()
    if del_resp["status"] == "error":
        raise gr.Error(del_resp["issue"], duration=None)
    
    return gr.Dataframe(headers=["API", "Key"], value=None)


# Update embedding options based on currently chosen type
def update_embedding_opts(embedding_type):
    global embeddings_dict
    new_opts = embeddings_dict[embedding_type.lower()]
    return gr.Dropdown(new_opts, value=new_opts[0])


# Update context options
def update_context_opts(no_context):
    if no_context:
        new_opts = [gr.Slider(visible=False), gr.Checkbox(visible=False)]
    else:
        new_opts = [gr.Slider(visible=True), gr.Checkbox(visible=True)]
    
    return new_opts


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
def run_rag(message, history, cur_topic, use_history, embedding_choice, model_choice, no_context, num_docs, chain_of_agents, chunk_size, chunk_overlap, refresh_db, uploaded_history):
    """Ensure the RAG system uses the desired setup, then request an answer from the system.

    Args:
      message: Current query to send to the RAG system.
      history: OpenAI style of conversation history for this session.
      cur_topic: Which topic directory to use
      use_history: Whether to use summary of chat history in query response.
      embedding_choice: Currently chosen embedding model to use.
      model_choice: Currently chosen chat model to use.
      no_context: Whether the RAG system should be used.
      num_docs: Number of chunks to use when creating an answer.
      chain_of_agents: Whether to use a chain of agents to summarize context.
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
    resp = requests.post(f"http://{rag_ip}:{rag_port}/setup/{cur_topic}", data=setup_info)
    setup_response = resp.json()
    if setup_response["status"] == "error":
        return setup_response['issue']

    if use_history and history:
        # Summarize history
        hist_info = {"user_history": history}
        resp = requests.post(f"http://{rag_ip}:{rag_port}/response/{cur_topic}", data=hist_info)
        hist_resp = resp.json()
        if hist_resp["status"] == "error":
            return hist_resp["issue"]
        hist_summary = hist_resp["response"]

        # Send query and history summary
        resp_info = {"user_query": message, "user_history": hist_summary, "no_context": no_context, "chain_of_agents": chain_of_agents}
        resp = requests.post(f"http://{rag_ip}:{rag_port}/response/{cur_topic}", data=resp_info)
        response_dict = resp.json()
        if response_dict["status"] == "error":
            return response_dict["issue"]
        response = response_dict["response"]
        context_text = response_dict["context_text"]
    else:
        # Send query to RAG system
        resp_info = {"user_query": message, "no_context": no_context, "chain_of_agents": chain_of_agents}
        resp = requests.post(f"http://{rag_ip}:{rag_port}/response/{cur_topic}", data=resp_info)
        response_dict = resp.json()
        if response_dict["status"] == "error":
            return response_dict["issue"]
        response = response_dict["response"]
        context_text = response_dict["context_text"]

    output = []
    output.append({"role": "assistant", "metadata": None, "content": response, "options": None})
    if context_text is not None:
        context_meta_info = {"title": "Context", "status": "done"}
        output.append({"role": "assistant", "metadata": context_meta_info, "content": context_text, "options": None})

    return output


# Get customization options, to allow storing/reloading UI theme
def get_setup_vars():
    """Create variables to pass to UI Block object.

    Returns:
      CSS, color option, theme object, chat layout, and saved css options to give to UI Block object.
    """
    # Add custom css
    css = """
    footer {visibility: hidden}
    """

    saved_avatar_size = "Small"
    try:
        with open("./customization/avatar_size.txt", "r", encoding="utf-8") as inf:
            saved_avatar_size = inf.readline().strip()
    except:
        pass
    saved_avatar_shape = "Circle"
    try:
        with open("./customization/avatar_shape.txt", "r", encoding="utf-8") as inf:
            saved_avatar_shape = inf.readline().strip()
    except:
        pass
    saved_avatar_fill = "Cover"
    try:
        with open("./customization/avatar_fill.txt", "r", encoding="utf-8") as inf:
            saved_avatar_fill = inf.readline().strip()
    except:
        pass

    sizes = {"small": 35, "medium": 75, "large": 115}

    cur_fill = saved_avatar_fill.lower()
    cur_shape = saved_avatar_shape.lower()
    cur_size = saved_avatar_size.lower()
    cur_base_size = sizes[cur_size]

    if cur_shape == "portrait":
        cur_width = cur_base_size
        cur_height = (cur_base_size + 30)
    elif cur_shape == "landscape":
        cur_width = (cur_base_size + 30)
        cur_height = cur_base_size
    else:
        cur_width = cur_base_size
        cur_height = cur_base_size
    if cur_shape == "circle":
        cur_border = "50%"
    else:
        cur_border = "10px"

    css += "\n.avatar-container {\n"
    css += f"    width: {cur_width}px !important;\n"
    css += f"    height: {cur_height}px !important;\n"
    css += f"    border-radius: {cur_border} !important;\n"
    css += "    padding: 0px !important;\n"
    css += "    flex-shrink: 0 !important;\n"
    css += "    bottom: 0 !important;\n"
    css += "    border: 1px solid var(--border-color-primary) !important;\n"
    css += "    overflow: hidden !important;\n"
    css += "}\n"

    css += "\n.avatar-image {\n"
    css += f"    width: {cur_width}px !important;\n"
    css += f"    height: {cur_height}px !important;\n"
    css += f"    border-radius: {cur_border} !important;\n"
    css += f"    object-fit: {cur_fill};\n"
    css += "    padding: 0px !important;\n"
    css += "    flex-shrink: 0 !important;\n"
    css += "    bottom: 0 !important;\n"
    css += "    border: 1px solid var(--border-color-primary) !important;\n"
    css += "    overflow: hidden !important;\n"
    css += "}\n"

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
        color_accent_soft='*primary_200',
        color_accent_soft_dark='*primary_700',
        border_color_primary='*primary_300',
        border_color_primary_dark='*primary_800',
        border_color_accent='*primary_300',
        border_color_accent_dark='*primary_950'
    )
    # color_accent_soft='*primary_700',
    # border_color_primary='*primary_800',
    # border_color_accent='*primary_950',

    cur_layout = "bubble"
    try:
        with open("./customization/chat_layout.txt", "r") as inf:
            cur_layout = inf.readline().strip()
    except:
        pass

    return (css, saved_color, theme, cur_layout, saved_avatar_size, saved_avatar_shape, saved_avatar_fill)


# Layout for the UI
def setup_layout(css, saved_color, theme, cur_layout, saved_avatar_size, saved_avatar_shape, saved_avatar_fill):
    """Create Block object to be used for UI.

    Args:
      css: String with custom CSS.
      saved_color: String naming the saved color.
      theme: Customized Default theme.
      cur_layout: String naming which layout to use for the chatbot.
      saved_avatar_size: String naming the saved avatar size.
      saved_avatar_shape: String naming the saved avatar shape.
      saved_avatar_fill: String naming the saved avatar fill.
    
    Returns:
      The Block object to be used for the UI.
    """
    ensure_customization()
    saved_user_icon = get_saved_user_icon()
    saved_chatbot_icon = get_saved_chatbot_icon()
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
                    use_history = gr.Checkbox(
                        label="Add summary of chat history to prompt",
                        value=False
                    )
                    no_context = gr.Checkbox(
                        value=False,
                        label="Do not use RAG system",
                        show_label=False
                    )
                    num_docs = gr.Slider(
                        1,
                        30,
                        value=5,
                        step=1,
                        label="Number of chunks to use when answering query",
                        visible=True
                    )
                    chain_of_agents = gr.Checkbox(
                        label="Use Chain of Agents to summarize each chunk",
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
                    initial_avatar_images = (f"./customization/user_icons/{saved_user_icon}", f"./customization/chatbot_icons/{saved_chatbot_icon}")
                    main_chat = gr.ChatInterface(
                        run_rag,
                        type="messages",
                        chatbot=gr.Chatbot(
                            type="messages",
                            show_label=False,
                            height=420,
                            avatar_images=initial_avatar_images,
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
                            no_context,
                            num_docs,
                            chain_of_agents,
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

                # Tab containing API key info
                with gr.Tab(label="Manage API Keys") as api_tab:
                    with gr.Row(equal_height=True):
                        save_keys = gr.Button(
                            value="Save API Keys",
                            variant="primary"
                        )
                        purge_keys = gr.Button(
                            value="Purge API Keys",
                            variant="stop"
                        )
                    api_keys = gr.Dataframe(
                        headers=["API", "Key"],
                        show_label=False,
                        interactive=True
                    )

        # Customization options
        with gr.Sidebar(width=300, open=False, position="right"):
            # User icon settings
            with gr.Group():
                user_icon_opts = get_user_icon_opts()
                user_icon_file = gr.Dropdown(
                    user_icon_opts,
                    value=saved_user_icon,
                    label="User Icon"
                )
                user_icon_preview = gr.Image(
                    value=f"./customization/user_icons/{saved_user_icon}",
                    image_mode=None,
                    type="filepath",
                    show_download_button=False,
                    show_share_button=False,
                    show_label=False,
                    interactive=False
                )
                new_user_icons = gr.Files(
                    show_label=False,
                    height=65
                )
                upload_user_icons = gr.Button(
                    value="Upload New User Avatar"
                )
                delete_user_icons = gr.Button(
                    value="Delete All User Avatars (Except Default)",
                    variant="stop"
                )

            # Chatbot icon settings
            with gr.Group():
                chatbot_icon_opts = get_chatbot_icon_opts()
                chatbot_icon_file = gr.Dropdown(
                    chatbot_icon_opts,
                    value=saved_chatbot_icon,
                    label="Chatbot Icon"
                )
                chatbot_icon_preview = gr.Image(
                    value=f"./customization/chatbot_icons/{saved_chatbot_icon}",
                    image_mode=None,
                    type="filepath",
                    show_download_button=False,
                    show_share_button=False,
                    show_label=False,
                    interactive=False
                )
                new_chatbot_icons = gr.Files(
                    show_label=False,
                    height=65
                )
                upload_chatbot_icons = gr.Button(
                    value="Upload New Chatbot Avatar"
                )
                delete_chatbot_icons = gr.Button(
                    value="Delete All Chatbot Avatars (Except Default)",
                    variant="stop"
                )

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
                )
                avatar_size = gr.Dropdown(
                    ["Small", "Medium", "Large"],
                    value=saved_avatar_size,
                    label="Avatar Size"
                )
                avatar_shape = gr.Dropdown(
                    ["Circle", "Square", "Portrait", "Landscape"],
                    value=saved_avatar_shape,
                    label="Avatar Shape"
                )
                avatar_fill = gr.Dropdown(
                    ["Cover", "Contain", "Fill"],
                    value=saved_avatar_fill,
                    label="Avatar Fill"
                )
                reload_app = gr.Button(
                    value="Reload UI\n(Requires refreshing tab)",
                    variant="stop"
                )

        # Handle API keys
        api_tab.select(show_api_keys, inputs=[cur_topic], outputs=[api_keys])
        save_keys.click(save_api_keys, inputs=[cur_topic, api_keys])
        purge_keys.click(purge_api_keys, inputs=[cur_topic], outputs=[api_keys])

        # Handle topics
        cur_topic.change(save_current_topic, inputs=[cur_topic])
        delete_topic.click(delete_current_topic, inputs=[cur_topic, embedding_choice], outputs=[cur_topic])
        make_topic.click(make_new_topic, inputs=[new_topic_name], outputs=[new_topic_name, cur_topic])

        # Handle context files
        context_tab.select(show_context_files, inputs=[cur_topic], outputs=[view_context_files])
        upload_context.click(context_to_server, inputs=[cur_topic, context_files], outputs=[context_files, view_context_files])
        download_all_context.click(dl_all_server_context, inputs=[cur_topic, view_context_files])
        view_context_files.delete(delete_single_context, inputs=[cur_topic])
        purge_context.click(delete_all_context, inputs=[cur_topic, view_context_files], outputs=[view_context_files])

        # Handle chat history
        save_history.click(history_to_local, inputs=[cur_topic, main_chat.chatbot, save_name])
        upload_history.click(local_to_history, inputs=[history_file], outputs=[view_history, main_chat.chatbot, history_file])

        # Handle general options
        embedding_type.change(update_embedding_opts, inputs=[embedding_type], outputs=[embedding_choice])
        model_type.change(update_chat_opts, inputs=[model_type], outputs=[model_choice])
        no_context.change(update_context_opts, inputs=[no_context], outputs=[num_docs, chain_of_agents])
        refresh_db.change(show_chunk_opts, inputs=[refresh_db], outputs=[chunk_size, chunk_overlap, reset_chunk_opts])
        reset_chunk_opts.click(chunk_opt_defaults, outputs=[chunk_size, chunk_overlap])

        # Handle customization options
        upload_user_icons.click(add_user_icons, inputs=[new_user_icons], outputs=[new_user_icons, user_icon_file])
        delete_user_icons.click(purge_user_icons, outputs=[user_icon_file])        
        upload_chatbot_icons.click(add_chatbot_icons, inputs=[new_chatbot_icons], outputs=[new_chatbot_icons, chatbot_icon_file])
        delete_chatbot_icons.click(purge_chatbot_icons, outputs=[chatbot_icon_file])        
        user_icon_file.change(update_user_ai_icons, inputs=[user_icon_file, chatbot_icon_file], outputs=[main_chat.chatbot, user_icon_preview, chatbot_icon_preview])
        chatbot_icon_file.change(update_user_ai_icons, inputs=[user_icon_file, chatbot_icon_file], outputs=[main_chat.chatbot, user_icon_preview, chatbot_icon_preview])
        chat_layout.change(update_chat_layout, inputs=[chat_layout], outputs=[main_chat.chatbot])
        mode_js = theme_mode_js()
        theme_mode.click(None, js=mode_js)
        theme_color.change(update_theme_color, inputs=[theme_color])
        avatar_size.change(update_avatar_size, inputs=[avatar_size])
        avatar_shape.change(update_avatar_shape, inputs=[avatar_shape])
        avatar_fill.change(update_avatar_fill, inputs=[avatar_fill])
        reload_app.click(update_reload)

    return app



if __name__ == "__main__":
    while True:
        # Launch UI with customization settings
        css, saved_color, theme, cur_layout, saved_avatar_size, saved_avatar_shape, saved_avatar_fill = get_setup_vars()
        app = setup_layout(css, saved_color, theme, cur_layout, saved_avatar_size, saved_avatar_shape, saved_avatar_fill)
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