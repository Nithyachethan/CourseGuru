#Streamlit app for chatbot with chat history and vectorstore
import streamlit as st
import os
import json
#from langchain import LLMChain
from langchain.chains import LLMChain
from langchain.vectorstores import VectorStore
from abc import ABC,abstractmethod
 
 
 
# Load utilities
from utils import Chat, StreamingStreamlitCallbackHandler, init_chat_llm, create_chat_prompt_template, load_vector_store
 
 
from abc import ABC, abstractmethod
 
class VectorStore(ABC):
    @abstractmethod
    def add_texts(self, texts):
        pass
 
    @abstractmethod
    def from_texts(self, texts):
        pass
 
    @abstractmethod
    def similarity_search(self, query, k):
        pass
 
# Create a subclass of VectorStore
class MyVectorStore(VectorStore):
    def __init__(self):
        # Initialize any necessary variables or data structures
        self.texts = []
        self.vectors = []
 
    def add_texts(self, texts):
        # Add new texts to the text store
        self.texts.extend(texts)
 
    def from_texts(self, texts):
        # Convert texts to vectors and store them
        self.vectors.extend([self._text_to_vector(text) for text in texts])
 
    def similarity_search(self, query, k):
        # Perform similarity search for the query
        query_vector = self._text_to_vector(query)
        # Placeholder logic: returning first k vectors for demonstration
        return self.vectors[:k]
 
    def _text_to_vector(self, text):
        # Placeholder method to convert text to vector
        # Implement your actual logic for vectorization here
        return [0.0] * 100  # Dummy vector of length 100
 
 
 
def parse_url_params():
    # Get the URL parameters
    params = st.experimental_get_query_params()
    #params = st.query_params
    # Parse the URL parameters to get the course and module
    course = params.get('course')[0] if 'course' in params else None
    return course
 
def display_chat_history():
    for query, response in st.session_state.chat.get_pipeline():
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            st.write(response)
 
def process_input(human_input, llm_supplier, model_name):
    vector_store=MyVectorStore()
    # Set the active query
    st.session_state.chat.set_active_query(human_input)
 
    vector_store.add_texts([human_input])
 
    # Extract document that are relevant to the query
    #relevant_documents = vector_store.similarity_search(human_input, k=3)
   
    relevant_documents = vector_store.similarity_search(human_input, k=3)
    print(relevant_documents)
    # Extract the relevant information from the documents
    chunks = [chunk.page_content for chunk in relevant_documents]
    page_numbers, titles, authors, file_paths = extract_metadata(relevant_documents)
 
    # Initialize the LLM
    llm = init_chat_llm(
        api_type=llm_supplier,
        model_name=model_name,
        #deployment_name="text-embedding-ada-002",
        callbacks=[StreamingStreamlitCallbackHandler(query_area=st.empty(), response_area=st.empty(), chat=st.session_state.chat)],
    )
        #model_name=model_name,
    # Run the LLM
    chain = LLMChain(llm=llm, prompt=chat_prompt_template)
    output = chain.run(human_input=human_input, relevant_documents="\n\n".join(chunks), chat_history=st.session_state.chat.history(config['chat_history_rounds']))
 
    # Add the dialogue to the chat history
    st.session_state.chat.append(human=human_input, assistant=output)
 
    # Display relevant documents in the sidebar
    show_document_sources(page_numbers, file_paths)
 
def extract_metadata(documents):
 
    # Extract the relevant information from the documents metadata
    page = [chunk.metadata["page"] for chunk in documents]
    titles = [chunk.metadata["title"] for chunk in documents]
    authors = [chunk.metadata["author"] for chunk in documents]
    file_paths = [chunk.metadata["file_path"] for chunk in documents]
 
    return page, titles, authors, file_paths
 
def show_document_sources(page, file_paths):
    st.sidebar.subheader(f"Further Reading")
    # Extract file names from file paths using os.path.basename
    file_names = [os.path.basename(path) for path in file_paths]
    # Zip and create a list of tuples (filename, page_number)
    combined = list(zip(file_names, page))    
    # Convert to set to remove duplicates
    unique_combined = set(combined)
    # Sort by filename first, then by page number
    sorted_combined = sorted(unique_combined, key=lambda x: (x[0], x[1]))
    # Create a list of strings combining filename and page number
    file_names_with_page = [f"{name} (p. {p + 1})" for name, p in sorted_combined]
    # Display the file paths
    for name, page in sorted_combined:
        # Encode spaces as %20 for URL compatibility
        encoded_name = name.replace(' ', '%20')
        # Create a link to the file
        link = f"{config['web_server_URL']}{encoded_name}#page={page + 1}"
        st.sidebar.markdown(f'<a href="{link}" target="_blank">{name} (p. {page + 1})</a>', unsafe_allow_html=True)
 
# def parse_url_params():
#     # Get the URL parameters
#     params = st.experimental_get_query_params()
   
#     # Parse the URL parameters to get the course and module
#     course = params.get('course')[0] if 'course' in params else None
#     return course
 
 
 
 
def get_subdirectories(directory):
    # Get a list of all items in the directory
    all_items = os.listdir(directory)
    # Filter out items that are not directories
    subdirs = [item for item in all_items if os.path.isdir(os.path.join(directory, item))]
    return subdirs
 
def valid(course, config):
 
    # Init paths
    paths = {
        "data": config['data_dir'],
        "course": None,
        "module": None,
        "upload": None,
        "download": None,
        "transcript": None,
        "summary": None,
    }
 
    # Ensure the data directory exists
    if not os.path.isdir(paths['data']):
        st.warning(f"Data directory {paths['data']} does not exist!")
        st.stop()
        return False
 
    # Check for course
    if not (course and course in get_subdirectories(paths['data'])):
        return False
 
    paths['course'] = os.path.join(paths['data'], course)
 
    return True
 
def set_course(config):
    # Init paths
    paths = {
        "data": config['data_dir'],
        "course": None,
        "module": None,
        "upload": None,
        "download": None,
        "transcript": None,
        "summary": None,
    }
 
    # Ensure the data directory exists
    if not os.path.isdir(paths['data']):
        st.warning(f"Data directory {paths['data']} does not exist!")
        return paths
 
    courses = ["Please select..."] + get_subdirectories(paths['data'])
    course = st.selectbox("Select Course", courses)
    # if course != "Please select...":
    #     paths['course'] = os.path.join(paths['data'], course)
    #     #if 'course' in st.query_params:
    #         #course = st.query_set_query_params['course'][0]
    #     st.query_params['course'] = course
    #     #else:
    #         #course = 'default_value'    
 
    if course != "Please select...":
        paths['course'] = os.path.join(paths['data'], course)
        st.experimental_set_query_params(course=course)
        #new_url = f"{st.experimental_set_query_params}?course={course}"
   
    # Update the browser's URL bar
        #st.rerun()
        #st.experimental_set_query_params(course=course)
       
        #course=st.query_params['course'][0]
 
 
 
def set_paths(course, config):
    data_dir = config['data_dir']
    paths = {
        "data": data_dir,
        "course": os.path.join(data_dir, course),
        "doc": os.path.join(data_dir, course, "doc"),
        "vectorstore": os.path.join(data_dir, course, "vector_store")
    }
    return paths
 
### START OF MAIN APP ###
 
# Import the JSON file 'config.json' as a dictionary
if "config" not in st.session_state:
    with open("./config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        print(config)
    st.session_state.config = config
config = st.session_state.config
 
# Initialize data & session
if "chat" not in st.session_state:
    st.session_state.chat = Chat()
 
# Set the page title and favicon
st.set_page_config(page_title=f"{config['app_name']}", page_icon=config['page_icon'], initial_sidebar_state="auto")
 
# Parse the URL parameters to get the course and module, if they exist
course = parse_url_params()
 
if valid(course, config):
 
    # Set the paths
    paths = set_paths(course, config)
 
    vector_store = load_vector_store(paths, config)
 
    chat_prompt_template = create_chat_prompt_template(paths, config)
 
    status_message = st.empty()
 
    # Initialize the sidebar
    st.sidebar.subheader("API Settings")
 
    # Select LLM Supplier
    llm_supplier = 'azure'
    #llm_supplier = st.sidebar.selectbox('Select LLM Supplier', ('azure', 'openai'))
 
    # Select model to use
    model_options = {
        "azure": ('GPT-4-Turbo', 'GPT4','gpt-35-turbo'),
        "openai": ('gpt-3.5-turbo', 'gpt-4')
    }
    model_name = st.sidebar.selectbox(f'Select Model for {llm_supplier.capitalize()}', model_options[llm_supplier])
    #model_name = 'gpt-35-turbo'
 
    # Display the app title and description
    st.title(f"{config['app_name']}: {course}")
    st.markdown(f"<small><i>Powered by {model_name} at {llm_supplier.capitalize()}</i></small>", unsafe_allow_html=True)
    st.write(f"{config['app_descr']}")
 
    # Display chat history
    if st.session_state.chat.is_pipeline():
        display_chat_history()
 
    # Display input area and get input
    max_words = config['max_words_per_query']
    input_area = st.chat_input(f"Enter your input (max {max_words} words):")
    if input_area:
        # Truncate the input to 200 words if necessary
        words = input_area.split()
        if len(words) > max_words:
            input_area = " ".join(words[:max_words])
            status_message.warning(f"Input truncated to {max_words} words.")
 
        # Process input
        process_input(input_area, llm_supplier=llm_supplier, model_name=model_name)
 
else:
    st.title(f"{config['app_name']}")
    st.write(f"{config['app_descr']}")
 
    # Set the course and module through selection boxes
    set_course(config)
    course = parse_url_params()
    if valid(course, config):
        st.button("Go")