# app/fine_tune.py
import json

def convert_docs_to_finetune_format(documents):
    qna_pairs = []
    for i, doc in enumerate(documents):
        qna_pairs.append({
            "prompt": f"Document context: {doc.page_content}\nQuestion: What is this section about?\nAnswer:",
            "completion": f" This section discusses: [Manual Answer for section {i}]."
        })
    with open("data/fine_tune_dataset.jsonl", "w") as f:
        for pair in qna_pairs:
            json.dump(pair, f)
            f.write("\n")
