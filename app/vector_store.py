# app/vector_store.py
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec
import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., aws-us-west-2
INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-chat-index")

cloud, region = PINECONE_ENV.split("-")[0], "-".join(PINECONE_ENV.split("-")[1:])
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region)
    )

index = pc.Index(INDEX_NAME)

def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    LangchainPinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    return LangchainPinecone(index, embeddings.embed_query, "text")

