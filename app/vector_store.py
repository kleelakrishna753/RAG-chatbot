# app/vector_store.py
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS

class MyEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

def create_vector_store(documents, persist_directory="db"):
    embeddings = MyEmbeddings()
    vectordb = FAISS.from_documents(documents, embeddings)
    return vectordb




