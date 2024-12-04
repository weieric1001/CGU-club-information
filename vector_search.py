import json
import os
import faiss
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def _average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_embeddings(input_texts):
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")

    batch_dict = tokenizer(
        input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )

    outputs = model(**batch_dict)
    embeddings = _average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings = embeddings.detach().numpy()
    return embeddings


def add_to_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def _vector_search(index, query_embedding, text_array, k=1):
    _, I = index.search(query_embedding, k)
    results = [text_array[i] for i in I[0]]
    scores = [{"score": round(float(1 - d), 4)} for d in _[0]]
    return [{**r, **s} for r, s in zip(results, scores)]


def vector_search(query: str) -> list[dict]:
    with open("社團問答v2.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    faiss_index = faiss.read_index("faiss_index.index")
    query_embedding = get_embeddings([f"query: {query}"])
    results = _vector_search(faiss_index, query_embedding, data, k=3)
    return results


def update_faiss_index():
    with open("社團問答v2.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Each input text should start with "query: " or "passage: ", even for non-English texts.
    # For tasks other than retrieval, you can simply use the "query: " prefix.
    input_texts = [f"query: {qa['question']}" for qa in data]
    embedding_array = get_embeddings(input_texts)
    faiss_index = add_to_faiss_index(embedding_array)
    faiss.write_index(faiss_index, "faiss_index.index")


if __name__ == "__main__":
    # update_faiss_index()

    with open("社團問答.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    results = vector_search("咖啡社")
    print(results)
