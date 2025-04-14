from RAG import *
from flask import Flask, request, jsonify

app = Flask(__name__)

vars = None
embedding = None
db = None
model = None

@app.route("/setup", methods=["POST"])
def setup():
    global vars, embedding, db, model

    # Get variables from frontend
    cur_vars = {"embedding_choice": None, "model_choice": None}
    cur_vars["embedding_choice"] = request.form["embedding_choice"]
    cur_vars["model_choice"] = request.form["model_choice"]

    # Check if initial setup should happen
    if vars is None:
        vars = get_vars(cur_vars["embedding_choice"], cur_vars["model_choice"])
        embedding = load_embedding(vars)
        db = load_database(vars, embedding)
        model = load_model(vars)
    else:
        # Check which options need updating
        updates = {"embedding_choice": False, "model_choice": False}
        for var in cur_vars.keys():
            if vars[var] != cur_vars[var]:
                updates[var] = True
        vars = get_vars(cur_vars["embedding_choice"], cur_vars["model_choice"])

        # Update embedding model, if needed
        if updates["embedding_choice"]:
            embedding = load_embedding(vars)
            db = load_database(vars, embedding)

        # Check if chat model needs updating
        if updates["model_choice"]:
            model = load_model(vars)
    
    return jsonify({"status": "ok"})


@app.route("/", methods=["POST"])
def main():
    global vars, db, model

    # Get needed variables
    local_model = vars["local_model"]
    model_choice = vars["model_choice"]

    # Get variables from frontend
    query = request.form["query"]

    # Get response from RAG system
    response = answer_query(vars, query, db, model)

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