# app/rag_chain.py
from transformers import pipeline
from huggingface_hub import login
import os

# Automatically login using HF token from environment
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token:
    login(token=hf_token)

class MistralQA:
    def __init__(self, retriever):
        self.retriever = retriever
        self.pipe = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")

    def run(self, query):
        context_docs = self.retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in context_docs])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        result = self.pipe(prompt, max_new_tokens=256, do_sample=True)[0]['generated_text']
        return result[len(prompt):].strip()

def get_qa_chain(vectordb):
    retriever = vectordb.as_retriever()
    return MistralQA(retriever)
