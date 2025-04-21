from RAG import *
from flask import Flask, request, jsonify, send_from_directory # type: ignore
from werkzeug.utils import secure_filename # type: ignore
from shutil import rmtree
import pandas as pd
import os, gc

app = Flask(__name__)

# Global variables to use
vars = None
embedding = None
db = None
model = None
cur_topic = None


@app.route("/get_api_keys/<topic>", methods=["POST"])
def get_api_keys(topic):
    # Get api root and file
    roots = get_vars(topic, only_roots=True)
    api_key_root = os.path.join(roots["API_ROOT"], "api_keys.csv")

    # Try making a new file if no file exists
    if not os.path.exists(api_key_root):
        try:
            with open(api_key_root, "w", encoding="utf-8") as outf:
                pass
            return jsonify({"status": "ok", "data": ""})
        except Exception as e:
            return jsonify({"status": "error", "issue": e.args[0]})
    
    # Try reading the file, using an empty string if the file is empty
    try:
        df = pd.read_csv(api_key_root, header=None)
        return jsonify({"status": "ok", "data": df.to_json()})
    except:
        df = ""
        return jsonify({"status": "ok", "data": df})


@app.route("/save_api_keys/<topic>", methods=["POST"])
def save_api_keys(topic):
    # Get api file path
    roots = get_vars(topic, only_roots=True)
    api_keys_root = roots["API_ROOT"]

    # Get new file
    file = request.files['file']

    # Remove old file
    try:
        os.remove(os.path.join(api_keys_root, "api_keys.csv"))
    except Exception as e:
        return jsonify({"staus": "error", "issue": e.args[0]})

    # Save new file
    try:
        file.save(os.path.join(api_keys_root, "api_keys.csv"))
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "issue": e.args[0]})


@app.route("/purge_api_keys/<topic>", methods=["POST"])
def purge_api_keys(topic):
    # Get api file path
    roots = get_vars(topic, only_roots=True)
    api_keys_path = os.path.join(roots["API_ROOT"], "api_keys.csv")

    try:
        with open(api_keys_path, "w", encoding="utf-8"):
            pass
    except Exception as e:
        return jsonify({"status": "error", "issue": e.args[0]})
    
    return jsonify({"status": "ok"})


@app.route("/delete_topic/<topic>", methods=["POST"])
def delete_topic(topic):
    global vars, embedding, db, cur_topic

    # Get data from request
    embedding_choice = request.form["embedding_choice"]

    # Get old topic root
    vars = get_vars(topic, embedding_choice=embedding_choice)
    topic_root = vars["roots"]["TOPIC_ROOT"]

    # Reset settings to Default topic
    embedding = load_embedding(vars)
    if db is not None:
        db._client.delete_collection(embedding_choice)
        db._client.clear_system_cache()
        db._client.reset()
        del db
        gc.collect()
        db = None
    cur_topic = "Default"

    # Remove old topic files
    try:
        rmtree(topic_root)
    except Exception as e:
        return jsonify({"status": "error", "issue": e.args[0]})
    
    return jsonify({"status": "ok"})


@app.route("/new_topic/<topic>", methods=["POST"])
def new_topic(topic):
    global vars, embedding, db, cur_topic

    # Set settings to new topic
    get_vars(topic, only_roots=True)
    db = None
    cur_topic = topic

    return jsonify({"status": "ok"})


@app.route("/context/<topic>", methods=["POST"])
def context(topic):
    global vars

    # Get needed variables
    roots = get_vars(topic, only_roots=True)

    # add all files to output
    output = []
    for file in os.listdir(roots["PDF_ROOT"]):
        output.append(file)
    for file in os.listdir(roots["CSV_ROOT"]):
        output.append(file)
    for file in os.listdir(roots["TXT_ROOT"]):
        output.append(file)
    
    return jsonify({'files': output})


@app.route("/download/<topic>/<name>")
def download_file(topic, name):
    global vars

    # Get needed variables
    roots = get_vars(topic, only_roots=True)

    # Determine directory file is in
    ext = name.split('.')[-1].strip().upper()
    root_dir = roots[f"{ext}_ROOT"]

    return send_from_directory(root_dir, name)


@app.route("/delete/<topic>/<name>", methods=["POST"])
def delete_file(topic, name):
    global vars

    # Get needed variables
    roots = get_vars(topic, only_roots=True)

    # Determine directory file is in    
    ext = name.split('.')[-1].strip().upper()
    root_dir = roots[f"{ext}_ROOT"]

    # Try deleting file
    try:
        os.remove(os.path.join(root_dir, name))

        with open(os.path.join(roots["CONTEXT_ROOT"], "last_modified.txt"), "w") as outf:
            outf.write(f"{time()}")

        return jsonify({'status': 'ok'})
    except:
        return jsonify({'status': 'error'})


@app.route("/upload/<topic>", methods=["POST"])
def upload_file(topic):
    global vars

    # Get needed variables
    roots = get_vars(topic, only_roots=True)

    # Get variables from frontend
    file = request.files['file']

    # Determine directory based on file type
    filename = secure_filename(file.filename)
    ext = filename.split('.')[-1].strip().upper()
    root_dir = roots[f"{ext}_ROOT"]

    # Try saving file
    try:
        file.save(os.path.join(root_dir, filename))

        with open(os.path.join(roots["CONTEXT_ROOT"], "last_modified.txt"), "w") as outf:
            outf.write(f"{time()}")

        return jsonify({'status': 'ok'})
    except:
        return jsonify({'status': 'error', 'file': filename})


@app.route("/setup/<topic>", methods=["POST"])
def setup(topic):
    global vars, embedding, db, model, cur_topic

    # Get variables from frontend
    cur_vars = {"embedding_choice": None, "model_choice": None}
    cur_vars["embedding_choice"] = request.form["embedding_choice"]
    cur_vars["model_choice"] = request.form["model_choice"]
    num_docs = int(request.form["num_docs"])
    chunk_size = int(request.form["chunk_size"])
    chunk_overlap = int(request.form["chunk_overlap"])
    refresh_db = request.form["refresh_db"]

    # Ensure variables are expected values
    if chunk_size < 1:
        chunk_size = 1
    if chunk_overlap < 0:
        chunk_overlap = 0
    if refresh_db.lower() == "true":
        refresh_db = True
    else:
        refresh_db = False

    # Check if initial setup should happen
    if (vars is None) or (embedding is None) or (db is None) or (model is None):
        try:
            vars = get_vars(topic, cur_vars["embedding_choice"], cur_vars["model_choice"], num_docs)
        except Exception as e:
            return jsonify({"status": "error", "issue": e.args[0]})
        vars["chunk_size"] = chunk_size
        vars["chunk_overlap"] = chunk_overlap
        vars["args"].refresh_db = refresh_db

        try:
            embedding = load_embedding(vars)
        except KeyError as e:
            return jsonify({"status": "error", "issue": f"No API key for {e.args[0]}"})
        except Exception as e:
            return jsonify({"status": "error", "issue": e})

        try:
            db = load_database(vars, embedding)
        except Exception as e:
            return jsonify({"status": "error", "issue": e})

        try:
            model = load_model(vars)
        except KeyError as e:
            return jsonify({"status": "error", "issue": f"No API key for {e.args[0]}"})
        except Exception as e:
            return jsonify({"status": "error", "issue": e})

        cur_topic = topic
    else:
        # Check which options need updating
        updates = {"embedding_choice": False, "model_choice": False}
        for var in cur_vars.keys():
            if vars[var] != cur_vars[var]:
                updates[var] = True
        
        try:
            vars = get_vars(topic, cur_vars["embedding_choice"], cur_vars["model_choice"], num_docs)
        except Exception as e:
            return jsonify({"status": "error", "issue": e.args[0]})

        # Update topic, if needed
        if cur_topic != topic:
            cur_topic = topic
            db = load_database(vars, embedding)

        # Update embedding model, if needed
        if updates["embedding_choice"]:
            try:
                embedding = load_embedding(vars)
            except KeyError as e:
                return jsonify({"status": "error", "issue": f"No API key for {e.args[0]}"})
            except Exception as e:
                return jsonify({"status": "error", "issue": e})

            try:
                db = load_database(vars, embedding)
            except Exception as e:
                return jsonify({"status": "error", "issue": e.args[0]})

        # Update database, if needed
        if refresh_db:
            vars["chunk_size"] = chunk_size
            vars["chunk_overlap"] = chunk_overlap
            vars["args"].refresh_db = refresh_db
            try:
                db = load_database(vars, embedding)
            except Exception as e:
                return jsonify({"status": "error", "issue": e.args[0]})

        # Update chat model, if needed
        if updates["model_choice"]:
            try:
                model = load_model(vars)
            except KeyError as e:
                return jsonify({"status": "error", "issue": f"No API key for {e.args[0]}"})
            except Exception as e:
                return jsonify({"status": "error", "issue": e})
    
    return jsonify({"status": "ok"})


@app.route("/response/<topic>", methods=["POST"])
def response(topic):
    global vars, db, model

    # Get needed variables
    roots = vars["roots"]
    local_model = vars["local_model"]
    model_choice = vars["model_choice"]

    # Determine if all context directories are empty
    context_roots = ["PDF_ROOT", "CSV_ROOT", "TXT_ROOT"]
    has_context = [bool(os.listdir(roots[root])) for root in context_roots]
    need_context = not any(has_context)
    if need_context:
        return jsonify({"response": "No context files have been provided, at least one is needed."})

    # Get variables from frontend
    try:
        user_query = request.form["user_query"]
    except:
        user_query = None
    try:
        user_history = request.form["user_history"]
    except:
        user_history = None
    try:
        chain_of_agents = request.form["chain_of_agents"]
    except:
        chain_of_agents = None

    # Get response from RAG system
    response = get_response(vars, db, model, user_query=user_query, user_history=user_history, chain_of_agents=chain_of_agents)

    # Format response
    if not isinstance(response, str):
        response = response.content
    if local_model:
        if (model_choice == "Mistral-7B-Instruct-v0.3"):
            prompt_end = response.find("[/INST]")
            response = response[(prompt_end + 7):]
        elif ("deepseek-r1" in model_choice):
            think_end = response.find("</think>")
            response = response[(think_end + 8):]

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run()