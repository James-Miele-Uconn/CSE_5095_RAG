"""Import statements."""

from huggingface_hub import notebook_login
from langchain_community.document_loaders import TextLoader, PyPDFDirectoryLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from time import time, gmtime, strftime
from sys import exit
from shutil import rmtree
import os, argparse


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

def get_vars():
    """Define setup variables.

    Returns:
      Dicitonary containing all variables to be used by other functions.
    """

    # Get arguments from command line
    args = parse_arguments()

    # Set directories to be used
    roots = dict()
    roots["EMBEDDING_ROOT"] = "D:\\Desktop\\AI\\Embeddings\\"
    roots["MODEL_ROOT"] = "D:\\Desktop\\AI\\LLMs\\"
    roots["PROJECT_ROOT"] = os.path.abspath(os.path.dirname(__file__))
    roots["CONTEXT_ROOT"] = os.path.join(roots["PROJECT_ROOT"], "context_files\\")
    roots["PDF_ROOT"] = os.path.join(roots["CONTEXT_ROOT"], "pdf_files\\")
    roots["CSV_ROOT"] = os.path.join(roots["CONTEXT_ROOT"], "csv_files\\")
    roots["CHROMA_ROOT"] = os.path.join(roots["PROJECT_ROOT"], "chroma_db_files\\")
    roots["MODIFIED_ROOT"] = os.path.join(roots["CHROMA_ROOT"], "(0)modified-times\\")
    roots["OUTPUT_ROOT"] = os.path.join(roots["PROJECT_ROOT"], "output_files\\")
    roots["API_ROOT"] = os.path.join(roots["PROJECT_ROOT"], "api_keys\\")

    # Create structural directories, if they don't exist
    ordered_roots = ["API_ROOT", "CHROMA_ROOT", "MODIFIED_ROOT", "CONTEXT_ROOT", "PDF_ROOT", "CSV_ROOT", "OUTPUT_ROOT"]
    for root in ordered_roots:
        if not os.path.exists(roots[root]):
            try:
                os.mkdir(roots[root])
            except Exception as e:
                print(f"Error making {root}:\n{e}")
                exit(1)

    # Determine if context directories are empty
    context_roots = ["PDF_ROOT", "CSV_ROOT"]
    need_context = False
    for root in context_roots:
        if not os.listdir(roots[root]):
            need_context = True

    # Check if context documents need to be added
    if need_context:
        print("\nYou have not supplied any documents to be used as context information.")
        print("Please do so before using this system.")
        exit(1)

    # Embedding to use, determines if running online
    embeddings_dict = {
        "ollama": ["nomic-embed-text", "mxbai-embed-large"],
        "local": ["nomic-embed-text-v1.5", "bert-base-uncased"],
        "online": ["openai"]
    }
    embedding_choice = args.embedding

    # Model to use, determines if running online
    # Models: deepseek-r1:[7b|14b|32b|70b], llama3.3, mistral, mixtral:8x7b
    models_dict = {
        "ollama": ["deepseek-r1:7b", "deepseek-r1:14b", "deepseek-r1:32b", "deepseek-r1:70b", "llama3.3", "mistral", "mixtral:8x7b", "deepseek-r1:671b"],
        "local": ["bert-base-uncased", "gpt2", "Mistral-7B-Instruct-v0.3", "zephyr-7b-beta", "DarkForest-20B-v3.0"],
        "online": ["openai"]
    }
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
            print("\nYou have chosen to use an online model but have not provided any api keys.")
            print("Please do so before using this system.")
            exit(1)

        with open(os.path.join(cur_root, os.listdir(cur_root)[0])) as inf:
            for line in inf:
                parts = line.strip().split(',')
                api_keys[parts[0]] = parts[1]
    
    output = {
        "args": args,
        "roots": roots,
        "embeddings_dict": embeddings_dict,
        "models_dict": models_dict,
        "embedding_choice": embedding_choice,
        "model_choice": model_choice,
        "local_embed": local_embed,
        "local_model": local_model,
        "api_keys": api_keys
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


def set_chroma_load(modified_times_loc, chroma_loc, cur_embedding, other_locs=list()):
    """Determine if Chroma should load from directory or start a new run.

    Args:
      modified_times_loc: Location of files saving last modified times for each embedding model.
      chroma_loc: Location of persistent directory for Chroma.
      cur_embedding: Current choice of embedding model, to check if a directory for said model exists.
      other_locs: List of locations with context docs to be checked for changes. Defaults to empty list.
    
    Returns:
      Boolean representing if Chroma should use saved files or create new files.
    """
    # Get last modified times for each of the directories holding context information
    context_times = [os.path.getmtime(folder) for folder in other_locs]

    # Get last modified time for specific embedding model, making file and setting time to 0 if none exists
    cur_embed_chroma_mod = None
    cur_embed_chroma_mod_loc = os.path.join(modified_times_loc, f"{cur_embedding}.txt")
    if not os.path.exists(cur_embed_chroma_mod_loc):
        # Make file with "time" of 0 if no file exists for this embedding model
        with open(cur_embed_chroma_mod_loc, "w") as outf:
            outf.write("0")
    with open(cur_embed_chroma_mod_loc, "r") as inf:
        for line in inf:
            cur_embed_chroma_mod = float(line.strip())
            break

    # Get booleans for determining if Chroma should load
    chroma_dir_exists = os.path.exists(os.path.join(chroma_loc, f"{cur_embedding}\\"))
    context_modified = any([mod > cur_embed_chroma_mod for mod in context_times])

    # Determine if Chroma should load
    if context_modified or not chroma_dir_exists:
        should_load = False
    else:
        should_load = True
    
    return should_load


def load_and_chunk(splitter, pdf_loc=None, csv_loc=None):
    """Load and chunk documents from specified locations.

    Args:
      splitter: The text splitter to use.
      pdf_loc: Path to directory containing pdf file(s).
      csv_loc: Path to directory containing csv file(s).
    
    Returns:
      Dictionary of chunked documents
    """
    # Load pdf documents
    pdf_chunks = None
    if pdf_loc:
      pdf_loader = PyPDFDirectoryLoader(pdf_loc)
      pdf_pages = pdf_loader.load()
      pdf_chunks = splitter.split_documents(pdf_pages)

    # Load csv documents
    csv_chunks = None
    if csv_loc:
      csv_loader = DirectoryLoader(csv_loc)
      csv_pages = csv_loader.load()
      csv_chunks = splitter.split_documents(csv_pages)

    output = {'pdf': None, 'csv': None, 'txt': None}
    if pdf_chunks:
      output['pdf'] = pdf_chunks
    if csv_chunks:
      output['csv'] = csv_chunks

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
    args = vars["args"]

    # Create directory for database using current embedding model, if needed
    cur_embed_db = os.path.join(roots["CHROMA_ROOT"], f"{embedding_choice}")
    if not os.path.exists(cur_embed_db):
        try:
            os.mkdir(cur_embed_db)
        except Exception as e:
            print(f"Error:\n{e}")
            exit(1)

    # Check if database should be loaded from saved files
    context_locs = [roots["PDF_ROOT"], roots["CSV_ROOT"]]
    if args.refresh_db:
        chroma_load = False
    else:
        chroma_load = set_chroma_load(roots["MODIFIED_ROOT"], roots["CHROMA_ROOT"], embedding_choice, other_locs=context_locs)

    # Set up ChromaDB based on whether or not pre-saved information should be used
    if chroma_load:
        db = Chroma(embedding_function=embedding, persist_directory=cur_embed_db)
    else:
        # Remove old folder
        try:
            rmtree(cur_embed_db)
        except Exception as e:
            print(f"Error:\n{e}")
            exit(1)
        
        # Create new folder
        try:
            os.mkdir(cur_embed_db)
        except Exception as e:
            print(f"Error:\n{e}")
            exit(1)

        # Give context information to Chroma
        # Not sure best way to handle, so create Chroma with first set of documents, then add any other documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = load_and_chunk(text_splitter, pdf_loc=roots["PDF_ROOT"], csv_loc=roots["CSV_ROOT"])
        db = Chroma.from_documents(chunks['pdf'], embedding, persist_directory=cur_embed_db)
        for key in chunks.keys():
            if key == "pdf":
                continue
            if chunks[key] is not None:
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


def answer_query(vars, query, db, model):
    """Given a query, create a prompt and receive a response.

    Args:
      vars: Dictionary containing setup variables.
      query: The query to answer.
      db: The colleciton of documents to use for RAG (assumes ChromaDB).
      model: Chat model to use.
    
    Returns:
      response received from the LLM model used
    """
    # Get needed variables
    args = vars["args"]
    model_choice = vars["model_choice"]
    models_dict = vars["models_dict"]

    # Set up context
    cur_chunks = db.similarity_search_with_score(query, k=args.num_docs)
    context_text = "\n\n".join([chunk.page_content for chunk, _score in cur_chunks])

    # Set up prompt
    prompt_template = """
    Answer the question based only on the following context:
    {context}
    Answer the question based on the above context: {question}.
    Add a new line after every sentence.
    Do not mention any information which is not contained within the context.
    """

    # Load context and query into prompt
    prompt_template = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt_template.format(context=context_text, question=query)

    # Get answer from LLM
    if (model_choice == "openai"):
        response = model.invoke(query)
    else:
        if (model_choice in models_dict["local"]):
            response = model.invoke(prompt)
        else:
            response = model.invoke(prompt)
    
    return response


def query_loop(vars, db, model):
    """Receive answer to a query, with ability to save to .txt file.

    Args:
      vars: Dictionary containing setup variables.
      db: Database of context information.
      model: Chat model to use.
    """
    # Get needed variables
    args = vars["args"]
    local_model = vars["local_model"]
    model_choice = vars["model_choice"]

    while True:
        # Receive response to query
        print("\n\nWelcome to the experimental RAG system! Enter a prompt, or 'exit' to exit.")
        print("\nHow can I help?\n")

        query = None
        try:
            query = input()
        except KeyboardInterrupt as e:
            if query is not None:
                pass
            else:
                break

        if query.lower() == "exit":
            break
        elif query == "":
            print("Please enter a query.")
            continue

        response = answer_query(vars, query, db, model)

        # Create output for question and response
        output = "\n"

        # Extract string of response, if needed
        if not isinstance(response, str):
            response = response.content

        # Add response to output
        if local_model:
            if (model_choice == "Mistral-7B-Instruct-v0.3"):
                prompt_end = response.find("[/INST]")
                output += response[(prompt_end + 7):]
            elif ("deepseek-r1" in model_choice):
                think_end = response.find("</think>")
                output += response[(think_end + 8):]
            else:
                output += response
        else:
            output += response

        # Print response for convenience
        print(output)

    print("\nThank you for using the RAG system!")
    return


def main():
    """Entry point for the program."""
    vars = get_vars()
    db, model = load_models_and_db(vars)
    query_loop(vars, db, model)


if __name__ == "__main__":
    main()