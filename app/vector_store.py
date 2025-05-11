# app/vector_store.py
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import Embedding

# Use SentenceTransformer model from Hugging Face or other available models
class SentenceTransformerEmbeddings(Embedding):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True, convert_to_tensor=True).tolist()

# Initialize SentenceTransformer Embedding
embeddings = SentenceTransformerEmbeddings()

# Use Chroma as vector store
vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)

