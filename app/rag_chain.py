# app/rag_chain.py
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

def get_qa_chain(vectordb):
    retriever = vectordb.as_retriever()
    llm = OpenAI(temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
