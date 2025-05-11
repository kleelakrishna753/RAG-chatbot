
# tests/test_vector_store.py
def test_vector_store_creation():
    from app.loader import load_and_split
    from app.vector_store import create_vector_store
    import os

    test_pdf_path = "data/sample.pdf"
    if not os.path.exists(test_pdf_path):
        print("Test PDF not found. Skipping test.")
        return

    docs = load_and_split(test_pdf_path)
    vectordb = create_vector_store(docs, persist_directory="test_db")
    assert vectordb is not None
    print("Vector store created successfully.")

