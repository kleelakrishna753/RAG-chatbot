# app/vector_store.py
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
import pinecone
import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., aws-us-west-2
INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-chat-index")

# Initialize old pinecone client (v2-style)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create index if it doesn't exist
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine"
    )

def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    LangchainPinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    return LangchainPinecone.from_existing_index(INDEX_NAME, embeddings)


