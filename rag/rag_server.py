from RAG import *
from flask import Flask, request, jsonify, send_from_directory # type: ignore
from werkzeug.utils import secure_filename # type: ignore
import os

app = Flask(__name__)

# Global variables to use
vars = None
embedding = None
db = None
model = None


@app.route("/context", methods=["POST"])
def context():
    global vars

    if vars is None:
        roots = get_vars(upload=True)
    else:
        roots = vars["roots"]

    output = []
    for file in os.listdir(roots["PDF_ROOT"]):
        output.append(file)
    for file in os.listdir(roots["CSV_ROOT"]):
        output.append(file)
    for file in os.listdir(roots["TXT_ROOT"]):
        output.append(file)
    
    return jsonify({'files': output})


@app.route("/download/<name>")
def download_file(name):
    global vars

    if vars is None:
        roots = get_vars(upload=True)
    else:
        roots = vars["roots"]

    ext = name.split('.')[-1].strip().upper()
    root_dir = roots[f"{ext}_ROOT"]

    return send_from_directory(root_dir, name)


@app.route("/delete/<name>", methods=["POST"])
def delete_file(name):
    global vars

    if vars is None:
        roots = get_vars(upload=True)
    else:
        roots = vars["roots"]
    
    ext = name.split('.')[-1].strip().upper()
    root_dir = roots[f"{ext}_ROOT"]

    try:
        os.remove(os.path.join(root_dir, name))
        return jsonify({'status': 'ok'})
    except:
        return jsonify({'status': 'error'})


@app.route("/upload", methods=["POST"])
def upload():
    global vars

    # Get needed variables
    if vars is None:
        roots = get_vars(upload=True)
    else:
        roots = vars["roots"]

    # Get variables from frontend
    file = request.files['file']

    # Determine directory based on file type
    filename = secure_filename(file.filename)
    ext = filename.split('.')[-1].strip().upper()
    root_dir = roots[f"{ext}_ROOT"]

    # Save file
    try:
        file.save(os.path.join(root_dir, filename))
        return jsonify({'status': 'ok'})
    except:
        return jsonify({'status': 'error', 'file': filename})


@app.route("/setup", methods=["POST"])
def setup():
    """Ensure RAG system setup uses desired settings."""
    global vars, embedding, db, model

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
    if vars is None:
        try:
            vars = get_vars(cur_vars["embedding_choice"], cur_vars["model_choice"], num_docs)
        except Exception as e:
            return jsonify({"status": "error", "issue": e.args[0]})

        vars["chunk_size"] = chunk_size
        vars["chunk_overlap"] = chunk_overlap
        vars["args"].refresh_db = refresh_db
        embedding = load_embedding(vars)
        db = load_database(vars, embedding)
        model = load_model(vars)
    else:
        # Check which options need updating
        updates = {"embedding_choice": False, "model_choice": False}
        for var in cur_vars.keys():
            if vars[var] != cur_vars[var]:
                updates[var] = True
        
        try:
            vars = get_vars(cur_vars["embedding_choice"], cur_vars["model_choice"], num_docs)
        except Exception as e:
            return jsonify({"status": "error", "issue": e.args[0]})

        # Update embedding model, if needed
        if updates["embedding_choice"]:
            embedding = load_embedding(vars)
            db = load_database(vars, embedding)

        # Update database, if needed
        if refresh_db:
            vars["chunk_size"] = chunk_size
            vars["chunk_overlap"] = chunk_overlap
            vars["args"].refresh_db = refresh_db
            db = load_database(vars, embedding)

        # Update chat model, if needed
        if updates["model_choice"]:
            model = load_model(vars)
    
    return jsonify({"status": "ok"})


@app.route("/response", methods=["POST"])
def response():
    """Get response from RAG system to given query."""
    global vars, db, model

    # Get needed variables
    local_model = vars["local_model"]
    model_choice = vars["model_choice"]

    # Get variables from frontend
    try:
        user_query = request.form["user_query"]
    except:
        user_query = None
    try:
        user_history = request.form["user_history"]
    except:
        user_history = None

    # Get response from RAG system
    response = get_response(vars, db, model, user_query=user_query, user_history=user_history)

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