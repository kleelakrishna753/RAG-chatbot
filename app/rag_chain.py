# app/rag_chain.py
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def get_qa_chain(vectorstore):
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # or any model you've downloaded
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)
    llm = HuggingFacePipeline(pipeline=pipe)

    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
