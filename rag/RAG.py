"""Import statements."""

from huggingface_hub import notebook_login # type: ignore
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFDirectoryLoader, DirectoryLoader # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # type: ignore
from langchain.prompts import ChatPromptTemplate # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore
from langchain_core.messages import HumanMessage, SystemMessage # type: ignore
from langchain_chroma import Chroma # type: ignore
from chromadb.config import Settings # type: ignore
from langchain_ollama import OllamaEmbeddings # type: ignore
from langchain_ollama import ChatOllama # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from time import time, gmtime, strftime
from sys import exit
import os, argparse, gc


def parse_arguments():
    """Parse command-line arguments using argparse.

    Returns:
      All arguments that can be set through the cmd.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", metavar="", help="\twhat embedding model to use. Default is mxbai-embed-large.", default="mxbai-embed-large", type=str)
    parser.add_argument("--model", metavar="", help="\twhat chat model to use. Default is deepseek-r1:32b", default="deepseek-r1:7b", type=str)
    parser.add_argument("--num_docs", metavar="", help="\thow many context chunks to use. Default is 5 chunks.", default=5, type=int)
    parser.add_argument("--refresh_db", help="\tforce the database to be remade.", action="store_true")
    args = parser.parse_args()

    return args


def get_vars(topic="Default", embedding_choice=None, model_choice=None, num_docs=None, only_roots=False):
    """Define setup variables.

    Args:
      topic: Which directory to use for context and database files.
      embedding_choice: Which embedding model to use. Defaults to None, to be overwritten.
      model_choice: Which chat model to use. Defaults to None, to be overwritten.
      num_docs: Number of chunks to use for answers. Defaults to None, as a flag for argparse.
      only_roots: Whether to only return roots. Defaults to False.

    Returns:
      Dicitonary containing all variables to be used by other functions.
    """
    # Get arguments from command line
    args = parse_arguments()

    # Modify num_docs if needed
    if num_docs is not None:
        args.num_docs = num_docs

    # Set directories to be used
    roots = dict()
    roots["EMBEDDING_ROOT"] = "D:\\Desktop\\AI\\Embeddings\\"
    roots["MODEL_ROOT"] = "D:\\Desktop\\AI\\LLMs\\"
    roots["SERVER_ROOT"] = os.path.abspath(os.path.dirname(__file__))
    roots["TOPIC_ROOT"] = os.path.join(roots["SERVER_ROOT"], f"{topic}\\")
    roots["CONTEXT_ROOT"] = os.path.join(roots["TOPIC_ROOT"], "context_files\\")
    roots["PDF_ROOT"] = os.path.join(roots["CONTEXT_ROOT"], "pdf_files\\")
    roots["CSV_ROOT"] = os.path.join(roots["CONTEXT_ROOT"], "csv_files\\")
    roots["TXT_ROOT"] = os.path.join(roots["CONTEXT_ROOT"], "txt_files\\")
    roots["CHROMA_ROOT"] = os.path.join(roots["TOPIC_ROOT"], "chroma_db_files\\")
    roots["MODIFIED_ROOT"] = os.path.join(roots["CHROMA_ROOT"], "(0)modified-times\\")
    roots["API_ROOT"] = os.path.join(roots["TOPIC_ROOT"], "api_keys\\")

    # Create structural directories, if they don't exist
    ordered_roots = [
        "TOPIC_ROOT",
        "API_ROOT",
        "CHROMA_ROOT",
        "MODIFIED_ROOT",
        "CONTEXT_ROOT",
        "PDF_ROOT",
        "CSV_ROOT",
        "TXT_ROOT"
    ]
    for root in ordered_roots:
        if not os.path.exists(roots[root]):
            try:
                os.mkdir(roots[root])
            except Exception as e:
                raise Exception(f"Could not create structural directory for {root}")

    # Exit early if only returning roots
    if only_roots:
        return roots

    # Embedding to use, determines if running online
    embeddings_dict = {
        "ollama": ["nomic-embed-text", "mxbai-embed-large"],
        "local": ["nomic-embed-text-v1.5", "bert-base-uncased"],
        "online": ["openai"]
    }
    if embedding_choice is None:
        embedding_choice = args.embedding

    # Model to use, determines if running online
    # Models: deepseek-r1:[7b|14b|32b|70b], llama3.3, mistral, mixtral:8x7b
    models_dict = {
        "ollama": ["deepseek-r1:7b", "deepseek-r1:14b", "deepseek-r1:32b", "deepseek-r1:70b", "llama3.3", "mistral", "mixtral:8x7b", "deepseek-r1:671b"],
        "local": ["bert-base-uncased", "gpt2", "Mistral-7B-Instruct-v0.3", "zephyr-7b-beta", "DarkForest-20B-v3.0"],
        "online": ["openai"]
    }
    if model_choice is None:
        model_choice = args.model

    # Flag to determine if embedding model is local
    local_embed = True
    if embedding_choice in embeddings_dict["online"]:
        local_embed = False

    # Flag to determine if chat model is local
    local_model = True
    if model_choice in models_dict["online"]:
        local_model = False

    # Setup api keys
    api_keys = dict()
    if (not local_embed) or (not local_model):
        # Check if a csv file containing api keys exists
        cur_root = roots["API_ROOT"]
        if not os.listdir(cur_root):
            raise Exception("No file containing API keys was provided.")

        with open(os.path.join(cur_root, os.listdir(cur_root)[0])) as inf:
            for line in inf:
                parts = line.strip().split(',')
                api_keys[parts[0]] = parts[1]
    
    chunk_size = 500
    chunk_overlap = 50

    output = {
        "args": args,
        "roots": roots,
        "embeddings_dict": embeddings_dict,
        "models_dict": models_dict,
        "embedding_choice": embedding_choice,
        "model_choice": model_choice,
        "local_embed": local_embed,
        "local_model": local_model,
        "api_keys": api_keys,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }

    return output


def load_embedding(vars):
    """Load embedding model.

    Args:
      vars: Dictionary containing setup variables.
    
    Returns:
      Embedding model to be used.
    """
    # Get needed variables
    local_embed = vars["local_embed"]
    embedding_choice = vars["embedding_choice"]
    embeddings_dict = vars["embeddings_dict"]
    roots = vars["roots"]
    api_keys = vars["api_keys"]

    # Create embedding based on variables
    if local_embed:
        if (embedding_choice in embeddings_dict["ollama"]):
            embedding = OllamaEmbeddings(model=embedding_choice)
        elif (embedding_choice in embeddings_dict["local"]):
            model_kwargs = {'trust_remote_code': True}
            embedding = HuggingFaceEmbeddings(model_name=f"{roots['EMBEDDING_ROOT']}{embedding_choice}\\", model_kwargs=model_kwargs)
    else:
        if (embedding_choice == "openai"):
            embedding = OpenAIEmbeddings(openai_api_key=api_keys["openai"])
    
    return embedding


def load_and_chunk(splitter, locs):
    """Load and chunk documents from specified locations.

    Args:
      splitter: The text splitter to use.
      locs: Dicitonary of paths to context file directories.
    
    Returns:
      Dictionary of chunked documents
    """
    # Load pdf documents
    pdf_chunks = None
    if os.listdir(locs['pdf_loc']):
      pdf_loader = PyPDFDirectoryLoader(locs['pdf_loc'])
      pdf_pages = pdf_loader.load()
      pdf_chunks = splitter.split_documents(pdf_pages)

    # Load csv documents
    csv_chunks = None
    if os.listdir(locs['csv_loc']):
      csv_loader = DirectoryLoader(locs['csv_loc'], loader_cls=CSVLoader)
      csv_pages = csv_loader.load()
      csv_chunks = splitter.split_documents(csv_pages)

    # Load txt documents
    txt_chunks = None
    if os.listdir(locs['txt_loc']):
        txt_loader = DirectoryLoader(locs['txt_loc'], loader_cls=TextLoader)
        txt_pages = txt_loader.load()
        txt_chunks = splitter.split_documents(txt_pages)

    output = {'pdf': pdf_chunks, 'csv': csv_chunks, 'txt': txt_chunks}

    return output


def load_database(vars, embedding):
    """Create or load database of context information.

    Args:
      vars: Dictionary containing setup variables.
      embedding: The embedding model to use for the database.
    
    Returns:
      Database to be used.
    """
    # Get needed variables
    roots = vars["roots"]
    embedding_choice = vars["embedding_choice"]
    need_refresh = vars["args"].refresh_db
    chunk_size = vars["chunk_size"]
    chunk_overlap = vars["chunk_overlap"]

    # Create directory for database using current embedding model, if needed
    cur_embed_db = os.path.join(roots["CHROMA_ROOT"], f"{embedding_choice}")
    if not os.path.exists(cur_embed_db):
        try:
            os.mkdir(cur_embed_db)
        except Exception as e:
            raise Exception(f"Error creating database directory for {embedding_choice}")

    # Determine last modified time of context information
    try:
        with open(f"{roots['CONTEXT_ROOT']}last_modified.txt", "r") as inf:
            cur_context_time = float(inf.readline().strip())
    except:
        raise Exception("Cannot make database as no context information is provided")

    # Determine last modified time of current embedding's database
    try:
        with open(f"{roots['MODIFIED_ROOT']}{embedding_choice}.txt", "r") as inf:
            cur_db_time = float(inf.readline().strip())
    except:
        cur_db_time = 0

    # Determine if database is out of sync with context information
    if cur_db_time < cur_context_time:
        need_refresh = True

    # Load or build database
    db = Chroma(collection_name=embedding_choice, embedding_function=embedding, persist_directory=cur_embed_db, client_settings=Settings(allow_reset=True))
    if need_refresh:
        # Remove old database info
        try:
            db._client.delete_collection(embedding_choice)
            db._client.clear_system_cache()
            db._client.reset()
            del db
            gc.collect()
            db = Chroma(collection_name=embedding_choice, embedding_function=embedding, persist_directory=cur_embed_db, client_settings=Settings(allow_reset=True))
        except:
            raise Exception(f"Error deleting and remaking database for {embedding_choice}")
        
        # Give context information to database
        context_locs = {
            'pdf_loc': roots["PDF_ROOT"],
            'csv_loc': roots["CSV_ROOT"],
            'txt_loc': roots["TXT_ROOT"]
        }
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = load_and_chunk(text_splitter, context_locs)
        for key in chunks.keys():
            # Ensure batch size is less than max size (currently 5461)
            if chunks[key] is not None:
                if len(chunks[key]) > 5461:
                    for idx in range(0, len(chunks[key]), 5461):
                        db.add_documents(documents=chunks[key][idx:idx+5461])
                else:
                    db.add_documents(documents=chunks[key])
        
        # Save current time as last modified time for context information for this embedding
        with open(f"{roots['MODIFIED_ROOT']}{embedding_choice}.txt", "w") as outf:
            outf.write(f"{time()}")

    return db


def load_model(vars):
    """Load chat model.

    Args:
      vars: Dictionary containing setup variables.
    
    Returns:
      Chat model to be used.
    """
    # Get needed variables
    local_model = vars["local_model"]
    model_choice = vars["model_choice"]
    models_dict = vars["models_dict"]
    roots = vars["roots"]
    api_keys = vars["api_keys"]

    if local_model:
        if (model_choice in models_dict["ollama"]):
            model = ChatOllama(model=model_choice)
        elif (model_choice in models_dict["local"]):
            llm = HuggingFacePipeline.from_model_id(model_id=f"{roots['MODEL_ROOT']}{model_choice}\\", task="text-generation", device=0)
            model = ChatHuggingFace(llm=llm)
    else:
        if (model_choice == "openai"):
            model = ChatOpenAI(openai_api_key=api_keys["openai"])
    
    return model


def load_models_and_db(vars):
    """Load embedding model and chat model, and load or create database.
    
    Args:
      vars: Dictionary containing setup variables.
    
    Returns:
      Tuple of database and chat model.
    """
    embedding = load_embedding(vars)
    db = load_database(vars, embedding)
    model = load_model(vars)

    return (db, model)


def get_response(vars, db, model, user_query=None, user_history=None, chain_of_agents=False):
    """Given input and some desired response type, create a prompt and receive a response.

    Args:
      vars: Dictionary containing setup variables.
      db: The colleciton of documents to use for RAG (assumes ChromaDB).
      model: Chat model to use.
      user_query: The query to give to the chat model. Defaults to None.
      user_history: The history to give to the chat model. Defaults to False.
      chain_of_agents: Whether to use a chain of agents to summarize context.
    
    Returns:
      Response received from the LLM model used
    """
    # Get needed variables
    args = vars["args"]
    model_choice = vars["model_choice"]
    models_dict = vars["models_dict"]

    context_text = None
    if (user_query is not None) and (user_history is not None):
        # Get context information
        cur_chunks = db.similarity_search_with_score(user_query, k=args.num_docs)
        context_chunks = [chunk.page_content for chunk, _score in cur_chunks]

        # Create context text
        if chain_of_agents:
            cur_summary = None
            for chunk in context_chunks:
                # Create prompt
                if cur_summary is None:
                    prompt_template = """
                    Create a summary of the following chunk of context information:
                    {context}
                    Give a detailed summary.
                    """
                    prompt_template = ChatPromptTemplate.from_template(prompt_template)
                    prompt = prompt_template.format(context=chunk)
                else:
                    prompt_template = """
                    Below is a summary of the previous context information used:
                    {summary}
                    Using the above information, create a summary of the following chunk of context information:
                    {context}
                    Give a detailed summary.
                    """
                    prompt_template = ChatPromptTemplate.from_template(prompt_template)
                    prompt = prompt_template.format(summary=cur_summary, context=chunk)

                # Get response
                if (model_choice == "openai"):
                    response = model.invoke(prompt)
                else:
                    if (model_choice in models_dict["local"]):
                        response = model.invoke(prompt)
                    else:
                        response = model.invoke(prompt)
                response = response.content

                # Format response
                if (model_choice == "Mistral-7B-Instruct-v0.3"):
                    prompt_end = response.find("[/INST]")
                    response = response[(prompt_end + 7):]
                elif ("deepseek-r1" in model_choice):
                    think_end = response.find("</think>")
                    response = response[(think_end + 8):]
                cur_summary = response

            chunk_text = "\n\n".join(context_chunks)
            context_text = f"Summary:\n{cur_summary}\n\nChunks:\n{chunk_text}"
        else:
            context_text = "\n\n".join(context_chunks)

        # Set up prompt
        prompt_template = """
        The following is a summary of the chat history so far:
        {history}
        Using the information above and the following context information, answer the question:
        {context}
        Answer the question based on the above information: {question}
        Do not mention anything that is not contained in either the context information or the chat history.
        Give a detailed response.
        """

        # Load history, context, and query into prompt
        prompt_template = ChatPromptTemplate.from_template(prompt_template)
        prompt = prompt_template.format(history=user_history, context=context_text, question=user_query)
    elif user_query is not None:
        # Get context information
        cur_chunks = db.similarity_search_with_score(user_query, k=args.num_docs)
        context_chunks = [chunk.page_content for chunk, _score in cur_chunks]

        # Create context text
        if chain_of_agents:
            cur_summary = None
            for chunk in context_chunks:
                # Create prompt
                if cur_summary is None:
                    prompt_template = """
                    Create a summary of the following chunk of context information:
                    {context}
                    Give a detailed summary.
                    """
                    prompt_template = ChatPromptTemplate.from_template(prompt_template)
                    prompt = prompt_template.format(context=chunk)
                else:
                    prompt_template = """
                    Below is a summary of the previous context information used:
                    {summary}
                    Using the above information, create a summary of the following chunk of context information:
                    {context}
                    Give a detailed summary.
                    """
                    prompt_template = ChatPromptTemplate.from_template(prompt_template)
                    prompt = prompt_template.format(summary=cur_summary, context=chunk)

                # Get response
                if (model_choice == "openai"):
                    response = model.invoke(prompt)
                else:
                    if (model_choice in models_dict["local"]):
                        response = model.invoke(prompt)
                    else:
                        response = model.invoke(prompt)
                response = response.content

                # Format response
                if (model_choice == "Mistral-7B-Instruct-v0.3"):
                    prompt_end = response.find("[/INST]")
                    response = response[(prompt_end + 7):]
                elif ("deepseek-r1" in model_choice):
                    think_end = response.find("</think>")
                    response = response[(think_end + 8):]
                cur_summary = response

            chunk_text = "\n\n".join(context_chunks)
            context_text = f"Summary:\n{cur_summary}\n\nChunks:\n{chunk_text}"
        else:
            context_text = "\n\n".join(context_chunks)

        # Set up prompt
        prompt_template = """
        Answer the question based only on the following context:
        {context}
        Answer the question based on the above context: {question}
        Do not mention any information which is not contained within the context.
        Give a detailed response.
        """

        # Load context and query into prompt
        prompt_template = ChatPromptTemplate.from_template(prompt_template)
        prompt = prompt_template.format(context=context_text, question=user_query)
    elif user_history is not None:
        # Set up prompt
        prompt_template = """
        Create a summary of the following chat history between a chatbot and a user:
        {history}
        """

        # Load history into prompt
        prompt_template = ChatPromptTemplate.from_template(prompt_template)
        prompt = prompt_template.format(history=user_history)
    else:
        return "This action requires either a query or a chat history."

    # Get answer from LLM
    if (model_choice == "openai"):
        response = model.invoke(prompt)
    else:
        if (model_choice in models_dict["local"]):
            response = model.invoke(prompt)
        else:
            response = model.invoke(prompt)
    
    if not isinstance(response, str):
        response = response.content
    
    metadata = None
    if context_text is not None:
        metadata = context_text
    return {"response": response, "metadata": metadata}