"""
Embedder module for transforming text chunks into vector embeddings using Hugging Face models.
"""

from typing import List, Dict, Tuple
import os
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load Hugging Face token from secrets if available
hf_token = st.secrets.get("huggingface", {}).get("token", None)

# Load the embedding model
@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5", use_auth_token=hf_token)

model = load_model()

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Generate vector embeddings for a list of text chunks.

    Args:
        texts: A list of strings (chunks)

    Returns:
        A NumPy array of shape (len(texts), embedding_dim)
    """
    return np.array(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))

def build_index(
    chunks: List[Dict[str, str]],
    existing_index=None,
    docs_meta: Dict[str, List[Dict]] = None,
    file_name: str = "default"
) -> Tuple[faiss.IndexFlatIP, Dict[str, List[Dict]]]:
    """
    Build or update a FAISS index from text chunks.

    Args:
        chunks: List of dicts with 'text' key
        existing_index: Existing FAISS index to update (optional)
        docs_meta: Metadata associated with the index (optional)
        file_name: Document name for metadata

    Returns:
        Updated FAISS index and metadata dictionary
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    if existing_index is None:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
    else:
        index = existing_index

    index.add(embeddings)

    if docs_meta is None:
        docs_meta = {}

    if file_name not in docs_meta:
        docs_meta[file_name] = []

    for i, chunk in enumerate(chunks):
        docs_meta[file_name].append({
            "chunk_id": i,
            "text": chunk["text"],
            "page": chunk.get("page", 1)
        })

    return index, docs_meta

def search(
    index: faiss.IndexFlatIP,
    query: str,
    docs_meta: Dict[str, List[Dict]],
    file_name: str,
    top_k: int = 5
) -> List[Dict[str, str]]:
    """
    Perform semantic search over indexed document chunks.

    Args:
        index: A FAISS index with document embeddings
        query: The user search query
        docs_meta: Metadata associated with document chunks
        file_name: Target document
        top_k: Number of top matches to return

    Returns:
        List of matched chunks with score and page info
    """
    query_vector = embed_texts([query])
    scores, indices = index.search(query_vector, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(docs_meta[file_name]):
            chunk_meta = docs_meta[file_name][idx]
            results.append({
                "score": round(float(score), 3),
                "page": chunk_meta["page"],
                "excerpt": chunk_meta["text"][:300] + "..." if len(chunk_meta["text"]) > 300 else chunk_meta["text"]
            })

    return results
