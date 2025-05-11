# app/vector_store.py
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set up the sentence transformer model
def create_vector_store(documents, persist_directory="db"):
    model_name = "all-MiniLM-L6-v2"  # lightweight, fast, and free
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb


