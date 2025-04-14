from RAG import *
from flask import Flask, request, jsonify

app = Flask(__name__)

vars = get_vars()
db, model = load_models_and_db(vars)

@app.route("/", methods=["POST"])
def main():
    global vars, db, model

    # Get needed variables
    local_model = vars["local_model"]
    model_choice = vars["model_choice"]

    # Get response from RAG system
    query = request.form["query"]
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