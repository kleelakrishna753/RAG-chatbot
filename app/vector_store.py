# app/vector_store.py
import numpy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_store(documents, persist_directory="db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb




