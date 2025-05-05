"""
Embedding module for generating and searching document embeddings.

This module provides functionality to generate embeddings using BGE and
perform similarity search using FAISS.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch

# Initialize the BGE model lazily to avoid Streamlit watcher issues
_model = None

def get_model() -> SentenceTransformer:
    """Get or initialize the BGE model."""
    global _model
    if _model is None:
        _model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return _model

def build_index(
    chunks: List[Dict[str, Any]],
    existing_index: Optional[faiss.Index] = None,
    existing_meta: Optional[Dict[str, Any]] = None,
    doc_name: Optional[str] = None
) -> Tuple[faiss.Index, Dict[str, Any]]:
    """
    Build or update a FAISS index with document chunks.
    
    Args:
        chunks: List of document chunks
        existing_index: Optional existing FAISS index
        existing_meta: Optional existing metadata
        doc_name: Name of the document being processed
        
    Returns:
        Tuple of (FAISS index, metadata dictionary)
    """
    # Generate embeddings
    texts = [chunk["text"] for chunk in chunks]
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    # Initialize or update metadata
    if existing_meta is None:
        meta = {"chunks": []}
    else:
        meta = existing_meta.copy()
        if "chunks" not in meta:
            meta["chunks"] = []
    
    # Add new chunks to metadata
    for i, chunk in enumerate(chunks):
        meta["chunks"].append({
            "text": chunk["text"],
            "page": chunk["page"],
            "metadata": chunk["metadata"]
        })
    
    if doc_name:
        meta["doc_names"] = set()
        meta["doc_names"].add(doc_name)
    
    # Initialize or update FAISS index
    if existing_index is None:
        # Create new index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
    else:
        # Update existing index
        index = existing_index
        index.add(embeddings)
    
    return index, meta

def search(
    index: faiss.Index,
    query: str,
    meta: Dict[str, Any],
    doc_name: Optional[str] = None,
    top_k: int = 8
) -> List[Dict[str, Any]]:
    """
    Perform semantic search using the FAISS index.
    
    Args:
        index: FAISS index
        query: Search query
        meta: Metadata dictionary
        doc_name: Optional document name to filter results
        top_k: Number of results to return
        
    Returns:
        List of search results, each containing:
        - text: The chunk text
        - page: Page number
        - score: Similarity score
    """
    # Generate query embedding
    model = get_model()
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Search
    scores, indices = index.search(query_embedding, top_k)
    
    # Format results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= len(meta["chunks"]):  # Safety check
            continue
            
        chunk = meta["chunks"][idx]
        
        # Filter by document name if specified
        if doc_name and chunk["metadata"]["source"] != doc_name:
            continue
            
        results.append({
            "text": chunk["text"],
            "page": chunk["page"],
            "score": float(score),
            "metadata": chunk["metadata"]
        })
    
    return results 