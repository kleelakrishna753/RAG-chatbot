from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., aws-us-west-2
INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-chat-index")

cloud, region = PINECONE_ENV.split("-")[0], "-".join(PINECONE_ENV.split("-")[1:])
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region)
    )else:
    # Index exists, safely delete vectors if needed
    index = pc.Index(INDEX_NAME)
    try:
        index.delete(delete_all=True)
    except Exception as e:
        print(f"Error deleting vectors: {e}")

index = pc.Index(INDEX_NAME)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def create_vector_store(documents):
    texts = [doc.page_content for doc in documents]
    embeddings = embedder.encode(texts).tolist()

    # Clear index for simplicity
    index.delete(delete_all=True)

    vectors = [{"id": f"doc-{i}", "values": emb, "metadata": {"text": text}} for i, (emb, text) in enumerate(zip(embeddings, texts))]
    index.upsert(vectors=vectors)
    return index

def query_vector_store(index, query_text, top_k=3):
    query_embedding = embedder.encode([query_text])[0].tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [match['metadata']['text'] for match in results['matches']]

