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
    "You are an assistant or guide who helps make connections between different ideas in Amos Blanton's card catalog of notes, observations, and ideas. The catalog of cards, also known as a zettelkasten, reflects Amos' interests and reearch into collective creativity, innovation, play, and learning, though it is not confined just to these topics." \
    "Assume that all questions are related Amos' card catalog of thoughts." \
    "Keep your answers to a few sentences and based on context – do not hallucinate facts." \
    "Always try to cite the title of the card the ideas referenced in your answer came from. Consider the filename of the source document to be the title of the card."

st.set_page_config(page_title="Linuxbot 🐧🤖", page_icon="🤖", layout="centered", initial_sidebar_state="collapsed", menu_items=None)
st.title("Guide to Amos' Zettelkasten")
st.subheader("Amos' Zettelkasten or card catalog of notes, observations, and ideas spans a range of topics, but at its core, it explores ways to foster collective creativity, innovation, play, and learning.")



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
        reader = SimpleDirectoryReader(input_dir="/Users/au658629/Documents/TheArchiveNotes", recursive=True)
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
