# https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/
import os
import streamlit as st

from llama_index.core import ServiceContext, Document, SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'localhost')
print(f"Connecting to ollama server {OLLAMA_HOST}")

# Connect to ollama service running on OpenShift
my_llm = "mistral:7b-instruct-v0.3-q8_0"

ollama_llm = Ollama(model=my_llm, base_url="http://"+OLLAMA_HOST+":11434")
ollama_embedding = OllamaEmbedding(
    model_name="mxbai-embed-large",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

system_prompt = \
    "You are an assistant for readers of the book titled The Years of Rice and Salt, by Kim Stanley Robinson. You have access to the full text of the book." \
    "The Years of Rice and Salt is an epic novel of alternate history that spans 2000 years. It explores themes of reincarnation. Major characters die and return again, but can be identified by the first letters of their names. "
    "Assume that all questions are related to the book The Years of Rice and Salt." \
    "Keep your answers based on context, and do not hallucinate facts." \
    "Always try to include a relevant passage from the text of the book with your answers."

st.set_page_config(page_title="Linuxbot 🐧🤖", page_icon="🤖", layout="centered", initial_sidebar_state="collapsed", menu_items=None)
st.title("Guide to Amos' Zettelkasten")
st.subheader("The Years of Rice and Salt Reader's Assistant")



with st.sidebar.expander("Settings"):
    system_prompt = st.text_area('System Prompt', value=system_prompt, height=256)
    #my_llm = st.text_area('Model', value=my_llm)

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Amos's catalog of thoughts."}
    ]

@st.cache_resource(show_spinner=False)
def load_data(_llm):
    with st.spinner(text="Loading and indexing the document data – might take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="../source-docs", recursive=True)
        docs = reader.load_data()
        
        Settings.llm = ollama_llm
        Settings.embed_model = ollama_embedding
        index = VectorStoreIndex.from_documents(docs)

        return index


index = load_data(ollama_llm)

chat_engine = index.as_chat_engine(
    chat_mode="context", verbose=True, system_prompt=system_prompt
)

if prompt := st.chat_input("Ask me a question about the catalog"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Querying..."):
            streaming_response = chat_engine.stream_chat(prompt)
            placeholder = st.empty()
            full_response = ''
            for token in streaming_response.response_gen:
                full_response += token
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message) # Add response to message history
