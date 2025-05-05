"""
Document loader module for processing PDF and DOCX files.

This module provides functionality to load and chunk documents from various formats
into a format suitable for embedding and analysis.
"""

from typing import List, Dict, Any, BinaryIO
import PyPDF2
from docx import Document
from pathlib import Path

def load_document(file: BinaryIO) -> List[Dict[str, Any]]:
    """
    Load and chunk a document from a file object.
    
    Args:
        file: A file-like object containing the document
        
    Returns:
        List of document chunks, each containing:
        - text: The chunk text
        - page: Page number
        - metadata: Additional metadata
    """
    file_ext = Path(file.name).suffix.lower()
    
    if file_ext == ".pdf":
        return _load_pdf(file)
    elif file_ext == ".docx":
        return _load_docx(file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def _load_pdf(file: BinaryIO) -> List[Dict[str, Any]]:
    """Load and chunk a PDF file."""
    chunks = []
    pdf_reader = PyPDF2.PdfReader(file)
    
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        
        # Split into chunks (simple paragraph-based splitting)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        for i, para in enumerate(paragraphs):
            chunks.append({
                "text": para,
                "page": page_num + 1,
                "metadata": {
                    "chunk_id": f"p{page_num+1}_{i}",
                    "source": file.name
                }
            })
    
    return chunks

def _load_docx(file: BinaryIO) -> List[Dict[str, Any]]:
    """Load and chunk a DOCX file."""
    chunks = []
    doc = Document(file)
    
    current_page = 1
    current_chunk = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
            
        # Simple heuristic for page breaks
        if len(current_chunk) > 0 and len(" ".join(current_chunk)) > 1000:
            chunks.append({
                "text": " ".join(current_chunk),
                "page": current_page,
                "metadata": {
                    "chunk_id": f"p{current_page}",
                    "source": file.name
                }
            })
            current_chunk = []
            current_page += 1
            
        current_chunk.append(text)
    
    # Add the last chunk
    if current_chunk:
        chunks.append({
            "text": " ".join(current_chunk),
            "page": current_page,
            "metadata": {
                "chunk_id": f"p{current_page}",
                "source": file.name
            }
        })
    
    return chunks 