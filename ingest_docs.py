# Ingesting the documents
import pickle
import glob
import os
import time
import faiss
import json
import sys
from tqdm import tqdm
from langchain.docstore import InMemoryDocstore
from langchain.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores import VectorStore
 
def create_vectorstore_from_documents(documents, api_type="azure", model_name="text-embedding-ada-002"):
 
 
    if api_type == "openai":
       
        # Get key from environment variable
        try:
            key = os.environ['OPENAI_API_KEY']
        except KeyError:
            raise KeyError("Please make sure the OPENAI_API_KEY environment variable is defined!")
       
        # Initialize the embeddings model
        embeddings_model = OpenAIEmbeddings(openai_api_key=key)
        #embeddings_model = AzureOpenAIEmbeddings(openai_api_key=key)
        # Create vectorstore and embed the documents
        print(f"Ingesting documents with {api_type}...")
        vectorstore = FAISS.from_documents(documents, embeddings_model)
       
 
    if api_type == "azure":
 
        # Get config from json file
        api_config_file = "./azure_api_config.json"
       
        with open(api_config_file, "r") as f:
            apis = json.load(f)
           
        try:
            api_config = next((item for item in apis if item["model"] == model_name), None)
        except KeyError:
            raise KeyError(f"Please make sure the model {model_name} is defined in azure_api_config.json!")
 
        # Get key, version, and endpoint from environment variable
        try:
            key = api_config["key"]
            version = api_config["version"]
            endpoint = api_config["endpoint"]
        except KeyError:
            raise KeyError(f"Error reading API config file: {api_config_file}")
 
        # Initialize the embeddings model
       
        # embeddings_model =AzureOpenAIEmbeddings(
        #     model=model_name,
        #     deployment_name="text-embedding-ada-002",
        #     openai_api_type=api_type,
        #     azure_endpoint=endpoint,
        #     api_version=version,
        #     api_key=key,
        #     chunk_size=1,
        # )
 
        embeddings_model = OpenAIEmbeddings(
            model=model_name,
            openai_api_type=api_type,
            openai_api_base=endpoint,
            openai_api_version=version,
            openai_api_key=key,
            chunk_size=1,
        )
 
        embeddings_model=AzureOpenAIEmbeddings(openai_api_key=key)
 
        # Create an empty FAISS vectorstore
        embedding_size = 1536
       
        index = faiss.IndexFlatL2(embedding_size)
       
        vectorstores = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
       
       
        # Embed the documents, one by one
        for document in tqdm(documents, desc=f"Ingesting documents with {api_type}"):
            vectorstore= vectorstores.add_texts(texts=[document.page_content], metadatas=[document.metadata])
           
            # delay to avoid rate limiting
            time.sleep(0.1)
       
       
    # else:
    #     raise ValueError(f"Unknown api_type: {api_type}")
   
    print(vectorstore)
    return vectorstore
   
   
def write_chunks(chunks, dpath):
 
    file_path = os.path.join(dpath, "chunks.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        for i, chunk in enumerate(chunks):
            file.write(f"Chunk {i + 1}:\n")
            file.write(chunk.page_content)
            file.write("\n" + "-" * 50 + "\n")
    print("Chunks written to chunks.txt")
 
def get_subdirectories(directory):
    # Get a list of all items in the directory
    all_items = os.listdir(directory)
    # Filter out items that are not directories
    subdirs = [item for item in all_items if os.path.isdir(os.path.join(directory, item))]
    return subdirs
 
 
# Import the JSON file 'config.json' as a dictionary
with open("./config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
 
print(get_subdirectories('./data'))
course = sys.argv[1]
if course in get_subdirectories('./data'):
 
    # Confirm that the user wants to ingest documents
    answer = input(f"Confirm that you want to ingest documents for {course} (y/n): ")
    if answer != "y":
        sys.exit()
   
    # Paths and constants
    api_type="azure"
    model_name="text-embedding-ada-002"
    doc_path = os.path.join("./data", course, "doc")
    print(doc_path)
    vector_store = os.path.join("./data", course, "vector_store", config["vectorstore_filename"])
    print(vector_store)
   
 
    chunk_size = 1000
    chunk_overlap = 50
    separators = ["\n\n", "  \n", "\n", " ", ""]
 
    # Find all PDF files in the specified subdirectory
    pdf_files = glob.glob(f"{doc_path}/*.pdf")
 
    # Load the documents
    loaders = [PyMuPDFLoader(pdf_file) for pdf_file in pdf_files]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
 
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators = separators
        )
    chunks = text_splitter.split_documents(documents)
   
    # Write the chunks to a text file
    write_chunks(chunks, doc_path)
   
    # Embed the documents and create a vectorstore
    vectorstore = create_vectorstore_from_documents(documents=chunks, api_type=api_type, model_name=model_name)
   
   
    # Save vectorstore
    #try:
    with open(vector_store, "wb") as f:
            pickle.dump(vectorstore,f)
            #pickle.dump(chunks,f)
            print(f"Vectorstore saved for {course}")
            #print(f"Vectorstore saved for {course}:{chunks}")
    #except Exception as e:
        #print("Error occurred while pickling vectorstore:", e)
    f.close()
else:
    print(f"Invalid course: {course}")
    sys.exit()
sys.exit()
 
 
 
 
 
 
 
# embeddings_model = OpenAIEmbeddings(
        #     model=model_name,
        #     openai_api_type=api_type,
        #     openai_api_base=endpoint,
        #     openai_api_version=version,
        #     openai_api_key=key,
        #     chunk_size=1,
        # )
 
        #embedding_size=768
 
        #vector_store = os.path.join("./data\TIA301", course, "vector_store", "vectorstore.pkl")
 
    # vector_store_directory = os.path.join("./data", course, "vector_store")
    # vector_store_file = os.path.join(vector_store_directory, "vectorstore.pkl")
    # os.makedirs(vector_store_directory, exist_ok=True)
 
    #print("hi")
        # vectorstore.save_local("faiss_store")
        # FAISS.load_local("faiss_store", AzureOpenAIEmbeddings())