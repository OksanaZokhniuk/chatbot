import os
import tempfile
import streamlit as st
from openai import OpenAI

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)

from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="💬")

st.title("💬 Chat with Your Documents")

st.write(
    "Upload PDF, TXT, or DOCX files and ask questions about them."
)

# -------------------------
# API KEY
# -------------------------
openai_api_key = st.text_input(
    "OpenAI API Key",
    type="password"
)

if not openai_api_key:
    st.info("Please enter your OpenAI API key.")
    st.stop()

# -------------------------
# LLAMAINDEX SETTINGS
# -------------------------
Settings.llm = LlamaOpenAI(
    model="gpt-3.5-turbo",
    api_key=openai_api_key,
)

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=openai_api_key,
)

# -------------------------
# SESSION STATE
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

# -------------------------
# FILE UPLOADER
# -------------------------
uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

# -------------------------
# PROCESS DOCUMENTS
# -------------------------
if uploaded_files and st.button("Process Documents"):

    with st.spinner("Processing documents..."):

        with tempfile.TemporaryDirectory() as temp_dir:

            # Save uploaded files
            for uploaded_file in uploaded_files:

                file_path = os.path.join(
                    temp_dir,
                    uploaded_file.name
                )

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # Load documents
            documents = SimpleDirectoryReader(
                temp_dir
            ).load_data()

            # Create vector index
            index = VectorStoreIndex.from_documents(
                documents
            )

            # Create query engine
            st.session_state.query_engine = (
                index.as_query_engine(
                    similarity_top_k=3
                )
            )

    st.success("Documents processed successfully!")

# -------------------------
# DISPLAY CHAT HISTORY
# -------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------
# CHAT INPUT
# -------------------------
prompt = st.chat_input(
    "Ask a question about your documents"
)

if prompt:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if docs processed
    if st.session_state.query_engine is None:

        response = (
            "Please upload and process documents first."
        )

    else:

        with st.spinner("Thinking..."):

            result = (
                st.session_state
                .query_engine
                .query(prompt)
            )

            response = str(result)

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )
