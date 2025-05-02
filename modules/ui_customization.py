import gradio as gr # type: ignore
import os
from shutil import copy

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