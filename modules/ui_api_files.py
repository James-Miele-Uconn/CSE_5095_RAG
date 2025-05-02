import gradio as gr # type: ignore
import pandas as pd
import os, requests

# Show api keys
def show_api_keys(rag_ip, rag_port, cur_topic):
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
def save_api_keys(rag_ip, rag_port, cur_topic, api_keys):
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
def purge_api_keys(rag_ip, rag_port, cur_topic):
    resp = requests.post(f"http://{rag_ip}:{rag_port}/purge_api_keys/{cur_topic}")
    del_resp = resp.json()
    if del_resp["status"] == "error":
        raise gr.Error(del_resp["issue"], duration=None)
    
    return gr.Dataframe(headers=["API", "Key"], value=None)