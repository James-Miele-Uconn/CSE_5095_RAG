"""Import statements."""

from huggingface_hub import notebook_login
from langchain_community.document_loaders import TextLoader, PyPDFDirectoryLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from time import time, gmtime, strftime
from sys import exit
import os, argparse


def load_and_chunk(pdf_loc=None, csv_loc=None):
    """Load and chunk documents from specified locations, for use with ChromaDB RAG system.

    Args:
      pdf_loc: path to directory containing pdf file(s)
      csv_loc: path to directory containing csv file(s)
    
    Returns:
      list of documents to use with Chroma
    """

    # Set up splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # Load pdf documents
    pdf_chunks = None
    if pdf_loc:
      pdf_loader = PyPDFDirectoryLoader(pdf_loc)
      pdf_pages = pdf_loader.load()
      pdf_chunks = text_splitter.split_documents(pdf_pages)

    # Load csv documents
    csv_chunks = None
    if csv_loc:
      csv_loader = DirectoryLoader(csv_loc)
      csv_pages = csv_loader.load()
      csv_chunks = text_splitter.split_documents(csv_pages)

    output = []
    if pdf_chunks:
      output.append(pdf_chunks)
    if csv_chunks:
      output.append(csv_chunks)

    return output


def answer_query(query, database, num_docs):
    """Given a query, create a prompt and receive a response.

    Args:
      query: The query to answer.
      database: The colleciton of documents to use for RAG (assumes ChromaDB).
      num_docs: How many documents should be given as context information.
    
    Returns:
      response received from the LLM model used
    """

    # Set up context
    docs_chroma = database.similarity_search_with_score(query, k=num_docs)
    context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

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
        response = model.predict(query)
    else:
        if (model_choice in local_models):
            response = model.invoke(prompt)
        else:
            response = model.invoke(prompt)
    
    return response


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


def parse_arguments():
    """Parse command-line arguments using argparse.

    Returns:
      All arguments that can be set through the cmd.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", metavar="", help="\twhat embedding model to use. Default is mxbai-embed-large.", default="mxbai-embed-large", type=str)
    parser.add_argument("--model", metavar="", help="\twhat chat model to use. Default is deepseek-r1:32b", default="deepseek-r1:32b", type=str)
    parser.add_argument("--num_docs", metavar="", help="\thow many context chunks to use. Default is 5 chunks.", default=5, type=int)
    args = parser.parse_args()

    return args


"""Define variables."""

# Get arguments from command line
args = parse_arguments()

# File system navigation
EMBEDDING_ROOT = "D:\\Desktop\\AI\\Embeddings\\"
MODEL_ROOT = "D:\\Desktop\\AI\\LLMs\\"
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
CONTEXT_ROOT = os.path.join(PROJECT_ROOT, "context_files\\")
PDF_ROOT = os.path.join(CONTEXT_ROOT, "pdf_files\\")
CSV_ROOT = os.path.join(CONTEXT_ROOT, "csv_files\\")
CHROMA_ROOT = os.path.join(PROJECT_ROOT, "chroma_db_files\\")
MODIFIED_ROOT = os.path.join(CHROMA_ROOT, "(0)modified-times\\")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output_files\\")

# Create structural directories, if they don't exist
roots = [CONTEXT_ROOT, PDF_ROOT, CSV_ROOT, CHROMA_ROOT, MODIFIED_ROOT, OUTPUT_ROOT]
need_context = False
for root in roots:
    if not os.path.exists(root):
        try:
            os.mkdir(root)
            need_context = True
        except Exception as e:
            print(f"Error making {root}:\n{e}")
            exit(1)

# Determine if context directories are empty
context_roots = [PDF_ROOT, CSV_ROOT]
for root in context_roots:
    if not os.listdir(root):
        need_context = True

# Check if context documents need to be added
if need_context:
    print("\nYou have not supplied any documents to be used as context information.")
    print("Please do so before using this system.")
    exit(1)

# Embedding to use, determines if running online
ollama_embeddings = ["nomic-embed-text", "mxbai-embed-large"]
local_embeddings = ["nomic-embed-text-v1.5", "bert-base-uncased"]
online_embeddings = ["openai"]
embeddings_choice = args.embedding

# Model to use, determines if running online
# Models: deepseek-r1:[7b|14b|32b|70b], llama3.3, mistral, mixtral:8x7b
ollama_models = ["deepseek-r1:7b", "deepseek-r1:14b", "deepseek-r1:32b", "deepseek-r1:70b", "llama3.3", "mistral", "mixtral:8x7b", "deepseek-r1:671b"]  # Don't use 671b
local_models = ["bert-base-uncased", "gpt2", "Mistral-7B-Instruct-v0.3", "zephyr-7b-beta", "DarkForest-20B-v3.0"]
online_models = ["openai"]
model_choice = args.model

# Flag to determine if Chroma should load from persistent directory
context_locs = [PDF_ROOT, CSV_ROOT]
chroma_load = set_chroma_load(MODIFIED_ROOT, CHROMA_ROOT, embeddings_choice, other_locs=context_locs)

# Flag to determine if program is running locally or not
local = True
if ((embeddings_choice == "openai") or (model_choice == "openai")):
    local = False

# For api keys
OPENAI_KEY = "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"


"""Generate embeddings and manage vectors."""

# Set up ChromaDB path and embedding based on embeddings_choice
cur_embed_db = os.path.join(CHROMA_ROOT, f"{embeddings_choice}")
if not os.path.exists(cur_embed_db):
    try:
        os.mkdir(cur_embed_db)
    except Exception as e:
        print(f"Error:\n{e}")
        exit(1)
if local:
    if (embeddings_choice in ollama_embeddings):
        embeddings = OllamaEmbeddings(model=embeddings_choice)
    elif (embeddings_choice in local_embeddings):
        model_kwargs = {'trust_remote_code': True}
        embeddings = HuggingFaceEmbeddings(model_name=f"{EMBEDDING_ROOT}{embeddings_choice}\\", model_kwargs=model_kwargs)
else:
    if (embeddings_choice == "openai"):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

# Set up ChromaDB based on whether or not pre-saved information should be used
if chroma_load:
    db_chroma = Chroma(embedding_function=embeddings, persist_directory=cur_embed_db)
else:
    # Give context information to Chroma
    # Not sure best way to handle, so create Chroma with first set of documents, then add any other documents
    chunks = load_and_chunk(pdf_loc=PDF_ROOT, csv_loc=CSV_ROOT)
    db_chroma = Chroma.from_documents(chunks[0], embeddings, persist_directory=cur_embed_db)
    for i in range(1, len(chunks)):
        db_chroma.add_documents(documents=chunks[i])

    # Save current time as last modified time for context information for this embedding
    with open(f"{MODIFIED_ROOT}{embeddings_choice}.txt", "w") as outf:
        outf.write(f"{time()}")

    # Flip chroma_load to True, to allow rerunning this section without remaking Chroma database
    chroma_load = True

# Set up model based on model_choice
if local:
    if (model_choice in ollama_models):
        model = ChatOllama(model=model_choice)
    elif (model_choice in local_models):
        llm = HuggingFacePipeline.from_model_id(model_id=f"{MODEL_ROOT}{model_choice}\\", task="text-generation", device=0)
        model = ChatHuggingFace(llm=llm)
else:
    if (model_choice == "openai"):
        model = ChatOpenAI(openai_api_key=OPENAI_KEY)


"""Receive answer to a query, with ability to save to .txt file."""

while True:
    # Receive response to query
    print("\n\n\n\n\n\n\n\nWelcome to the experimental RAG system! Enter a prompt, or 'exit' to exit.")
    print("\nHow can I help?\n")
    query = input()

    if query.lower() == "exit":
        print("\nThank you for using the RAG system. Goodbye!")
        exit()

    response = answer_query(query, db_chroma, args.num_docs)

    # Create output for question and response
    output = "\n"

    # Extract string of response, if needed
    if not isinstance(response, str):
        response = response.content

    # Add response to output
    if local:
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