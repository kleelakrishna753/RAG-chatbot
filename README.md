---

# 📄 RAG-based Document Chatbot

This is a **Retrieval-Augmented Generation (RAG)** chatbot that allows users to interact with their uploaded documents by asking questions and getting intelligent answers. The project utilizes **Mistral 7B**, **HuggingFace**, and **PineconeDB** to provide contextually relevant responses based on the document content. The app is built with **Streamlit** for an easy-to-use web interface.

## 🚀 Features

* **Upload PDF documents**: Upload PDF files and split them into chunks for processing.
* **Index content using Pinecone**: The content of the document is indexed in **Pinecone** for fast, efficient search.
* **Interactive Q\&A**: Ask questions based on the document content, and get intelligent responses powered by **Mistral 7B**.
* **Easy deployment**: Built using **Streamlit**, it’s easy to deploy and run locally or on Streamlit Cloud.
* **Fine-tuning**: Bonus feature to fine-tune the model on custom Q\&A data for domain-specific use cases.

## 🧑‍💻 Tech Stack

* **LangChain**: For document processing and embeddings.
* **PineconeDB**: Vector database to store and retrieve document embeddings.
* **Mistral 7B**: Powerful large language model (LLM) for generating responses.
* **Streamlit**: Web framework for building interactive apps.
* **Hugging Face**: Provides access to pre-trained models like Mistral 7B.

## 🛠 Requirements

### Dependencies

* **langchain**: For document processing and vector store integration.
* **transformers**: For using pre-trained models from Hugging Face.
* **huggingface-hub**: For interacting with Hugging Face's model repository.
* **sentence-transformers**: For generating embeddings.
* **streamlit**: Framework for building interactive web apps.
* **pinecone-client**: For interacting with PineconeDB.
* **pypdf**: For reading PDF files.

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-repo/rag-chatbot.git
   cd rag-chatbot
   ```

2. **Create a Virtual Environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Environment Variables**:

   * Create a `.env` file in the root directory.

   * Add the following credentials:

     ```plaintext
     PINECONE_API_KEY=your_pinecone_api_key
     PINECONE_ENV=aws-us-west-2  # Replace with your Pinecone environment region
     PINECONE_INDEX=rag-chat-index  # The name of the index in Pinecone
     HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
     ```

   * You can get your **Pinecone API key** and **environment** from your Pinecone console.

   * You can get your **Hugging Face API token** from your [Hugging Face account](https://huggingface.co/settings/tokens).

5. **Login to Hugging Face CLI**:
   If you haven't logged in yet, use the following command to authenticate your Hugging Face CLI:

   ```bash
   huggingface-cli login
   ```

## 🔧 Running the App

1. **Start the Streamlit app**:

   ```bash
   streamlit run ui/app.py
   ```

2. The app will open in your default web browser. You can upload a PDF document and start asking questions based on the content of the document.

---

## 🎯 How It Works

1. **Document Upload**: Users can upload PDF documents through the Streamlit interface.
2. **Document Splitting**: The document is split into smaller chunks for easier processing and indexing.
3. **Embeddings**: The chunks are embedded into vector representations using a Hugging Face model (e.g., `sentence-transformers/all-MiniLM-L6-v2`).
4. **Indexing in Pinecone**: The embeddings are stored in **Pinecone**, which allows fast retrieval of relevant document chunks during querying.
5. **Querying**: Users can ask questions, and the chatbot retrieves relevant document chunks from Pinecone, then passes the context to **Mistral 7B** for generating an answer.
6. **Answer Generation**: **Mistral 7B** is used to generate natural language answers based on the retrieved context.

---

## 🧑‍💻 Fine-Tuning (Optional)

You can fine-tune the model with custom Q\&A pairs based on your documents.

### Steps to Fine-Tune:

1. **Convert Documents to Fine-Tune Format**:
   Use the provided script to convert your document content into question-answer pairs.

   ```python
   from app.loader import load_and_split
   from app.fine_tune import convert_docs_to_finetune_format

   docs = load_and_split("data/sample.pdf")
   convert_docs_to_finetune_format(docs)
   ```

2. **Train the Model**: Use the generated `fine_tune_dataset.jsonl` with Hugging Face or OpenAI tools to fine-tune the model on your dataset.

---

### Troubleshooting

* **Pinecone Index Not Found**: Make sure the index exists in Pinecone and that the credentials in `.env` are correct.
* **Invalid Hugging Face Token**: If you encounter the "Invalid user token" error, re-login to the Hugging Face CLI by running `huggingface-cli login`.
* **Streamlit Errors**: Ensure all required dependencies are installed, and check the logs for more details.

---

Feel free to contribute, open issues, or suggest improvements!

