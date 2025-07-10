import gradio as gr # type: ignore
import requests

# Main chat function
def run_rag(message, history, rag_ip, rag_port, history_uploaded, context_changed, cur_topic, use_history, embedding_choice, model_choice, no_context, num_pdfs, num_csvs, num_txts, chain_of_agents, chunk_size, chunk_overlap, refresh_db, uploaded_history):
    """Ensure the RAG system uses the desired setup, then request an answer from the system.

    Args:
      message: Current query to send to the RAG system.
      history: OpenAI style of conversation history for this session.
      rag_ip: IP address of RAG server.
      rag_port: Port of RAG server.
      history_uploaded: Whether a chat history was uploaded.
      context_changed: Whether the context files were changed since last run.
      cur_topic: Which topic directory to use
      use_history: Whether to use summary of chat history in query response.
      embedding_choice: Currently chosen embedding model to use.
      model_choice: Currently chosen chat model to use.
      no_context: Whether the RAG system should be used.
      num_pdfs: Number of pdf chunks to use when creating an answer.
      num_csvs: Number of csv chunks to use when creating an answer.
      num_txts: Number of txt chunks to use when creating an answer.
      chain_of_agents: Whether to use a chain of agents to summarize context.
      chunk_size: Size of chunks to use for database chunks.
      chunk_overlap: Amount of overlap to use for database chunks.
      refresh_db: Whether the database should be forcibly refreshed.
      uploaded_history: Uploaded history to add.
    
    Returns:
      Formatted string response to the given query.
    """
    # Add uploaded history to current history, if needed
    new_history_uploaded = gr.Checkbox()
    if history_uploaded:
        for idx in range(len(uploaded_history)):
            history.insert(idx, uploaded_history[idx])
        new_history_uploaded = gr.Checkbox(value=False)

    # Force database refresh if context information has changed
    new_context_changed = gr.Checkbox()
    if context_changed:
        refresh_db = True
        new_context_changed = gr.Checkbox(value=False)

    # Send info for RAG system setup
    setup_info = {
        "embedding_choice": embedding_choice,
        "model_choice": model_choice,
        "num_pdfs": num_pdfs,
        "num_csvs": num_csvs,
        "num_txts": num_txts,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "refresh_db": refresh_db
    }
    resp = requests.post(f"http://{rag_ip}:{rag_port}/setup/{cur_topic}", data=setup_info)
    setup_response = resp.json()
    if setup_response["status"] == "error":
        return setup_response['issue']

    if use_history and history:
        # Summarize history
        history = "\n\n".join([msg["content"] for msg in history])
        hist_info = {"user_history": history}
        resp = requests.post(f"http://{rag_ip}:{rag_port}/response/{cur_topic}", data=hist_info)
        hist_resp = resp.json()
        if hist_resp["status"] == "error":
            return hist_resp["issue"]
        hist_summary = hist_resp["response"]

        # Send query and history summary
        resp_info = {"user_query": message, "user_history": hist_summary, "no_context": no_context, "chain_of_agents": chain_of_agents}
        resp = requests.post(f"http://{rag_ip}:{rag_port}/response/{cur_topic}", data=resp_info)
        response_dict = resp.json()
        if response_dict["status"] == "error":
            return response_dict["issue"]
        response = response_dict["response"]
        context_text = response_dict["context_text"]
    else:
        # Send query to RAG system
        resp_info = {"user_query": message, "no_context": no_context, "chain_of_agents": chain_of_agents}
        resp = requests.post(f"http://{rag_ip}:{rag_port}/response/{cur_topic}", data=resp_info)
        response_dict = resp.json()
        if response_dict["status"] == "error":
            return response_dict["issue"]
        response = response_dict["response"]
        context_text = response_dict["context_text"]

    output = []
    output.append({"role": "assistant", "metadata": None, "content": response, "options": None})
    if use_history and history:
        history_meta_info = {"title": "History", "status": "done"}
        output.append({"role": "assistant", "metadata": history_meta_info, "content": hist_summary, "options": None})
    if context_text is not None:
        context_meta_info = {"title": "Context", "status": "done"}
        output.append({"role": "assistant", "metadata": context_meta_info, "content": context_text, "options": None})

    return [output, new_history_uploaded, new_context_changed]