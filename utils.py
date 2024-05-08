import streamlit as st
import pickle
import os
import json
import importlib.util
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import AzureChatOpenAI,ChatOpenAI
#from langchain import PromptTemplate, LLMChain
#from langchain.chat_models import ChatOpenAI,AzureChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain.vectorstores import VectorStore
 
# Define a class for storing the chat history
class Chat:
    def __init__(self):
        self.pipeline = []
        self.active_query = ""
 
    def append(self, human, assistant):
        self.pipeline.append((human, assistant))
 
    def set_active_query(self, query):
        self.active_query = query
 
    def get_active_query(self):
        return self.active_query
   
    def is_pipeline(self):
        return bool(self.pipeline)
   
    def get_pipeline(self):
        return self.pipeline
 
    def history(self, n):
        result = []
        for pair in self.pipeline[-n:]:
            result.append(f"Human: {pair[0]}\nAssistant: {pair[1]}")
        return "\n".join(result)
 
    def full_history_markdown(self):
        result = []
        for pair in self.pipeline:
            result.append(f"<b><i>{pair[0]}</i></b><br>{pair[1]}<br><hr>")
        return "".join(result)
 
# Define a callback handler for streaming stdout
class StreamingStreamlitCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self, query_area, response_area, chat: Chat):
        super().__init__()
        self.response_area = response_area
        self.query_area = query_area
        self.active_query = chat.get_active_query()
        self.active_response = ""
 
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if isinstance(token, str):
            with self.query_area.chat_message("user"):
                st.write(self.active_query)
            with self.response_area.chat_message("assistant"):
                self.active_response += token  # update the accumulated text
                st.write(self.active_response, unsafe_allow_html=True)  # write the accumulated text
 
# Function initializing the LLM, basedon provider (openai or azure)
def init_chat_llm(api_type, model_name, temperature=0, streaming=True, callbacks=[]):
 
    if api_type == "openai":
       
        # Get key from environment variable
        try:
            key = os.environ['OPENAI_API_KEY']
        except KeyError:
            raise KeyError("Please make sure the OPENAI_API_KEY environment variable is defined!")
       
        # Initialize
        llm = ChatOpenAI(
            openai_api_key=key,
            streaming=streaming,
            callbacks=callbacks,
            temperature=temperature,
            model_name=model_name,
           
        )
 
    elif api_type == "azure":
 
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
 
        # Initialize
        llm = AzureChatOpenAI(
            streaming=streaming,
            callbacks=callbacks,
            temperature=temperature,
            #deployment_name=model_name,
            model_name="",
            #deployment_name="gupol-text-embedding",
            openai_api_type=api_type,
            api_version=version,
            azure_endpoint=endpoint,
            api_key=key,
        )
 
        #openai_api_base=endpoint,
    else:
        raise ValueError(f"Unknown api_type: {api_type}")
 
    return llm
 
@st.cache_data
def create_chat_prompt_template(paths, config):
 
    # Load templates from course directory
    prompt_templates_file = os.path.join(paths['course'], config['prompt_templates_filename'])
    spec = importlib.util.spec_from_file_location("prompt_templates", prompt_templates_file)
    prompt_templates = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_templates)
 
    system_template = prompt_templates.system_template
    human_template = prompt_templates.human_template
 
    system_message_prompt_template = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt_template = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt_template, human_message_prompt_template])
    return chat_prompt_template
 
# @st.cache_data
# def vector_store_path_exists(paths,config):
#     path = os.path.join(paths['vectorstore'], config['vectorstore_filename'])
#     if not os.path.exists(path):
#         print("File does not exist:", path)
#     return None
 
# @st.cache_data
# def vector_store_getsize(paths,config):
#     path = os.path.join(paths['vectorstore'], config['vectorstore_filename'])
#     if os.path.getsize(path) == 0:
#         print("File is empty:", path)
#     return None
 
 
# @st.cache_data
# def load_vector_store(paths,config):
#     path = os.path.join(paths['vectorstore'], config['vectorstore_filename'])
#     print(path)
#     try:
#         with open(path, "rb") as f:
#             vector_store = pickle.load(f)
#             print(vector_store)
#         return vector_store  
#     except Exception as e:
#         print("Error occurred while unpickling vectorstore:", e)
   
@st.cache_data
def load_vector_store(paths,config):
    path = os.path.join(paths['vectorstore'], config['vectorstore_filename'])
    print(path)
    with open(path, "rb") as f:
        #global vectorstore
        #local_vectorstore: VectorStore = pickle.load(f)
        vector_store = pickle.load(f)
        print(vector_store)
    return vector_store