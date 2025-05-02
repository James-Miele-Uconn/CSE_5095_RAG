import gradio as gr # type: ignore
import os
from time import strftime, gmtime

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
    history = []
    try:
        with open(history_file, "r", encoding="utf-8") as inf:
            history = inf.readlines()
            if history:
                history = eval(history[0])
        new_history_uploaded = gr.Checkbox(value=True)
    except Exception as e:
        raise gr.Error(e, duration=None)

    output = [
        gr.Chatbot(type="messages", value=history),
        gr.Chatbot(type="messages", value=history),
        gr.File(value=None),
        new_history_uploaded
    ]

    return output