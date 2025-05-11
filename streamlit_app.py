# ui/app.py

import streamlit as st
from app.loader import load_and_split
from app.vector_store import create_vector_store
from app.rag_chain import get_qa_chain
import tempfile

st.set_page_config(page_title="📄 Chat with your Document")
st.title("📄 Chat with your Document")

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

