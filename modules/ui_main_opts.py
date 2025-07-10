import gradio as gr # type: ignore


# Options lists for each type of embedding model
embeddings_dict = {
    "ollama": ["mxbai-embed-large", "snowflake-arctic-embed:335m", "nomic-embed-text"],
    "local": ["nomic-embed-text-v1.5", "bert-base-uncased"],
    "online": ["openai"]
}

# Options lists for each type of chat model
models_dict = {
    "ollama": ["mistral:7b-instruct", "mistral:7b", "mistral:7b-instruct-fp16", "mixtral:8x7b", "deepseek-r1:7b", "deepseek-r1:14b", "deepseek-r1:32b", "deepseek-r1:70b", "r1-1776:70b", "llama3.3"],
    "local": ["bert-base-uncased", "gpt2", "Mistral-7B-Instruct-v0.3", "zephyr-7b-beta", "DarkSapling-7B-v2.0"],
    "online": ["openai"]
}


# Update embedding options based on currently chosen type
def update_embedding_opts(embedding_type):
    global embeddings_dict
    new_opts = embeddings_dict[embedding_type.lower()]
    return gr.Dropdown(new_opts, value=new_opts[0])


# Update model options based on currently chosen type
def update_chat_opts(model_type):
    global models_dict
    new_opts = models_dict[model_type.lower()]
    return gr.Dropdown(new_opts, value=new_opts[0])


# Update context options
def update_context_opts(no_context):
    if no_context:
        new_opts = [gr.Slider(visible=False), gr.Slider(visible=False), gr.Slider(visible=False), gr.Checkbox(visible=False)]
    else:
        new_opts = [gr.Slider(visible=True), gr.Slider(visible=True), gr.Slider(visible=True), gr.Checkbox(visible=True)]
    
    return new_opts


# Show or hide chunk options
def show_chunk_opts(rebuild_val):
    new_size = gr.Number(visible=rebuild_val)
    new_overlap = gr.Number(visible=rebuild_val)
    new_reset = gr.Button(visible=rebuild_val)

    return [new_size, new_overlap, new_reset]


# Reset chunk options to default values
def chunk_opt_defaults():
     return [gr.Number(value=500), gr.Number(value=50)]