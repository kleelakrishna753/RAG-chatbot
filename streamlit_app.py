import subprocess
import streamlit as st
import os
from app.loader import load_and_split
from app.vector_store import create_vector_store
from app.rag_chain import get_qa_chain
import tempfile

# Set page config
st.set_page_config(page_title="📄 Chat with your Document")
st.title("📄 Chat with your Document")

# Hugging Face login
st.sidebar.title("🔐 Hugging Face Login")
hf_token = st.sidebar.text_input("Enter your Hugging Face Token", type="password")

if st.sidebar.button("Login to Hugging Face"):
    if hf_token:
        try:
            result = subprocess.run(
                ["huggingface-cli", "login", "--token", hf_token],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            st.sidebar.success("✅ Logged in successfully.")
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        except subprocess.CalledProcessError as e:
            st.sidebar.error("❌ Login failed: " + e.stderr.strip())
    else:
        st.sidebar.warning("⚠️ Please enter a token.")


# Upload PDF section
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Processing document..."):
        docs = load_and_split(tmp_path)
        vectordb = create_vector_store(docs)
        qa_chain = get_qa_chain(vectordb)
        st.success("Ready to chat!")

    user_query = st.text_input("Ask a question about the document:")
    if user_query:
        with st.spinner("Thinking..."):
            result = qa_chain.run(user_query)
            st.markdown(f"**Answer:** {result}")

