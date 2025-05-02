import gradio as gr # type: ignore
import os, requests
from shutil import copy

# Load context files to rag server
def context_to_server(rag_ip, rag_port, cur_topic, files):
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
    
    new_context_changed = gr.Checkbox(True)
    return [gr.Files(value=None), gr.Files(value=show_context_files(rag_ip, rag_port, cur_topic)), new_context_changed]


# Show the currently uploaded context files
def show_context_files(rag_ip, rag_port, cur_topic):
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
def delete_single_context(rag_ip, rag_port, cur_topic, deleted: gr.DeletedFileData):
    fname = os.path.basename(deleted.file.path)
    resp = requests.post(f"http://{rag_ip}:{rag_port}/delete/{cur_topic}/{fname}")
    del_resp = resp.json()
    if del_resp['status'] == "error":
        raise gr.Error(f"Error deleting file {fname}", duration=None)
    
    new_context_changed = gr.Checkbox(value=True)
    
    return new_context_changed


# Delete all server context files
def delete_all_context(rag_ip, rag_port, cur_topic, files_list):
    if files_list is None:
        return gr.Files(value=None)

    for file in files_list:
        fname = os.path.basename(file)
        resp = requests.post(f"http://{rag_ip}:{rag_port}/delete/{cur_topic}/{fname}")
        del_resp = resp.json()
        if del_resp['status'] == "error":
            raise gr.Error(f"Error deleting file {fname}", duration=None)

    new_context_changed = gr.Checkbox(value=True)
    
    return [gr.Files(value=None), new_context_changed]