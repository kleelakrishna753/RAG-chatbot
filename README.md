# RAG-based Document Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot that lets you chat with your uploaded documents.

## Features
- Upload PDF documents
- Split, embed, and index content
- Ask questions and get intelligent answers
- Powered by LangChain, OpenAI, and ChromaDB
- Bonus: Fine-tune an LLM with your own document-based Q&A pairs

## Run the App
```bash
pip install -r requirements.txt
streamlit run ui/app.py
```

## Bonus: Fine-Tuning
Add Q&A examples by running:
```python
from app.loader import load_and_split
from app.fine_tune import convert_docs_to_finetune_format

docs = load_and_split("data/sample.pdf")
convert_docs_to_finetune_format(docs)
```
Use the generated `fine_tune_dataset.jsonl` with OpenAI or HuggingFace tools.

## License
MIT
