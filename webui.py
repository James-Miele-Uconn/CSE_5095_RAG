from time import sleep
import gradio as gr # type: ignore
import os, argparse
from modules.ui_context_files import *
from modules.ui_history import *
from modules.ui_topics import *
from modules.ui_api_files import *
from modules.ui_main_opts import *
from modules.ui_customization import *
from modules.main_rag import *


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


# Update global variable checking for reload
def update_reload():
    global need_restart
    need_restart = True


# Hack to allow downloading files from localhost
webui_args = parse_webui_arguments()
whitelist_ip = webui_args.rag_server_ip
gr.processing_utils.PUBLIC_HOSTNAME_WHITELIST.append(whitelist_ip)
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost, 127.0.0.1/8, ::1"

# Global value for tracking if webui server restart is needed
need_restart = False

# Get customization options, to allow storing/reloading UI theme
def get_setup_vars():
    """Create variables to pass to UI Block object.

    Returns:
      Argparse object, CSS, color option, theme object, chat layout, and saved css options to give to UI Block object.
    """
    # Get command-line arguments
    vars = parse_webui_arguments()

    # Add custom css
    css = """
    footer {visibility: hidden}
    """

    # Get saved avatar icon info
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
        if not os.path.exists("./customization"):
            try:
                os.mkdir("./customization")
            except:
                pass
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

    cur_layout = "bubble"
    try:
        with open("./customization/chat_layout.txt", "r") as inf:
            cur_layout = inf.readline().strip()
    except:
        pass

    return (vars, css, saved_color, theme, cur_layout, saved_avatar_size, saved_avatar_shape, saved_avatar_fill)


# Layout for the UI
def setup_layout(vars, css, saved_color, theme, cur_layout, saved_avatar_size, saved_avatar_shape, saved_avatar_fill):
    """Create Block object to be used for UI.

    Args:
      vars: Argparse object containing command-line arguments
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
    saved_user_icon = get_saved_user_icon()
    saved_chatbot_icon = get_saved_chatbot_icon()
    with gr.Blocks(title="RAG System", theme=theme, css=css) as app:
        rag_ip = gr.Markdown(value=vars.rag_server_ip, visible=False)
        rag_port = gr.Markdown(value=vars.rag_server_port, visible=False)
        history_uploaded = gr.Checkbox(value=False, visible=False)
        context_changed = gr.Checkbox(value=False, visible=False)

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
                with gr.Group():
                    use_history = gr.Checkbox(
                        label="Add summary of chat history to prompt",
                        value=False
                    )
                    no_context = gr.Checkbox(
                        value=False,
                        label="Do not use RAG system",
                        show_label=False
                    )
                    chain_of_agents = gr.Checkbox(
                        label="Use Chain of Agents to summarize each chunk",
                        value=False
                    )
                    with gr.Accordion(label="Chunk Options", open=False):
                        num_pdfs = gr.Slider(
                            1,
                            10,
                            value=3,
                            step=1,
                            label="Number of pdf chunks",
                            visible=True
                        )
                        num_csvs = gr.Slider(
                            1,
                            10,
                            value=3,
                            step=1,
                            label="Number of csv chunks",
                            visible=True
                        )
                        num_txts = gr.Slider(
                            1,
                            10,
                            value=3,
                            step=1,
                            label="Number of txt chunks",
                            visible=True
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
                            height=600,
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
                            rag_ip,
                            rag_port,
                            history_uploaded,
                            context_changed,
                            cur_topic,
                            use_history,
                            embedding_choice,
                            model_choice,
                            no_context,
                            num_pdfs,
                            num_csvs,
                            num_txts,
                            chain_of_agents,
                            chunk_size,
                            chunk_overlap,
                            refresh_db,
                            view_history
                        ],
                        additional_outputs=[
                            history_uploaded,
                            context_changed
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
        api_tab.select(show_api_keys, inputs=[rag_ip, rag_port, cur_topic], outputs=[api_keys])
        save_keys.click(save_api_keys, inputs=[rag_ip, rag_port, cur_topic, api_keys])
        purge_keys.click(purge_api_keys, inputs=[rag_ip, rag_port, cur_topic], outputs=[api_keys])

        # Handle topics
        cur_topic.change(save_current_topic, inputs=[cur_topic])
        delete_topic.click(delete_current_topic, inputs=[rag_ip, rag_port, cur_topic, embedding_choice], outputs=[cur_topic])
        make_topic.click(make_new_topic, inputs=[rag_ip, rag_port, new_topic_name], outputs=[new_topic_name, cur_topic])

        # Handle context files
        context_tab.select(show_context_files, inputs=[rag_ip, rag_port, cur_topic], outputs=[view_context_files])
        upload_context.click(context_to_server, inputs=[rag_ip, rag_port, cur_topic, context_files], outputs=[context_files, view_context_files, context_changed])
        download_all_context.click(dl_all_server_context, inputs=[cur_topic, view_context_files])
        view_context_files.delete(delete_single_context, inputs=[rag_ip, rag_port, cur_topic], outputs=[context_changed])
        purge_context.click(delete_all_context, inputs=[rag_ip, rag_port, cur_topic, view_context_files], outputs=[view_context_files, context_changed])

        # Handle chat history
        save_history.click(history_to_local, inputs=[cur_topic, main_chat.chatbot, save_name])
        upload_history.click(local_to_history, inputs=[history_file], outputs=[view_history, main_chat.chatbot, history_file, history_uploaded])

        # Handle general options
        embedding_type.change(update_embedding_opts, inputs=[embedding_type], outputs=[embedding_choice])
        model_type.change(update_chat_opts, inputs=[model_type], outputs=[model_choice])
        no_context.change(update_context_opts, inputs=[no_context], outputs=[num_pdfs, num_csvs, num_txts, chain_of_agents])
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
        vars, css, saved_color, theme, cur_layout, saved_avatar_size, saved_avatar_shape, saved_avatar_fill = get_setup_vars()
        app = setup_layout(vars, css, saved_color, theme, cur_layout, saved_avatar_size, saved_avatar_shape, saved_avatar_fill)
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