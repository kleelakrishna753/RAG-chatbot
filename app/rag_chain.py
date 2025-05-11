from transformers import pipeline
from huggingface_hub import login
import os
from app.vector_store import query_vector_store

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token:
    login(token=hf_token)

class MistralQA:
    def __init__(self, index):
        self.index = index
        self.pipe = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")

    def run(self, query):
        context_docs = query_vector_store(self.index, query)
        context = "\n".join(context_docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        result = self.pipe(prompt, max_new_tokens=256, do_sample=True)[0]['generated_text']
        return result[len(prompt):].strip()

def get_qa_chain(index):
    return MistralQA(index)

