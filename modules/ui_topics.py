import gradio as gr # type: ignore
import os, requests

# Get all topics
def get_all_topics():
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
    current_topic = "Default"
    try:
        with open("./customization/current_topic.txt") as inf:
            current_topic = inf.readline().strip()
    except:
        pass

    return current_topic


# Save current topic
def save_current_topic(cur_topic):
    try:
        with open("./customization/current_topic.txt", "w", encoding="utf-8") as outf:
            outf.write(f"{cur_topic}\n")
    except:
        pass


# Detele currently selected topic, other than Default
def delete_current_topic(rag_ip, rag_port, cur_topic, embedding_choice):
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
def make_new_topic(rag_ip, rag_port, new_topic):
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