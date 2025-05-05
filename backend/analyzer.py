"""
Analyzer module for processing NDA documents using DeepSeek.

This module provides functionality to analyze NDA documents using the DeepSeek
language model, extracting summaries, risk scores, and critical clauses.
"""

from typing import List, Dict, Any
import os
import json
import requests
from pathlib import Path
import streamlit as st

# Load prompts
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

def _load_prompt(name: str) -> str:
    """Load a prompt from the prompts directory."""
    prompt_path = PROMPTS_DIR / f"{name}.txt"
    with open(prompt_path, "r") as f:
        return f.read().strip()

def _call_deepseek(
    prompt: str,
    model_name: str = "deepseek-llm-67b-reasoning",
    temperature: float = 0.7
) -> str:
    """
    Call the DeepSeek API via Fireworks.
    
    Args:
        prompt: The prompt to send to the model
        model_name: The model to use
        temperature: Sampling temperature
        
    Returns:
        Model response as a string
    """
    api_key = st.secrets["fireworks"]["api_key"] if "fireworks" in st.secrets else os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY not set in secrets or .env")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": 2000
    }
    
    response = requests.post(
        "https://api.fireworks.ai/inference/v1/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        raise Exception(f"API call failed: {response.text}")
    
    return response.json()["choices"][0]["text"]

def run_full_analysis(
    chunks: List[Dict[str, Any]],
    model_name: str = "deepseek-llm-67b-reasoning",
    temperature: float = 0.7,
    top_k: int = 8
) -> Dict[str, Any]:
    """
    Run a full analysis of an NDA document.
    
    Args:
        chunks: List of document chunks
        model_name: Model to use for analysis
        temperature: Sampling temperature
        top_k: Number of chunks to use for analysis
        
    Returns:
        Dictionary containing:
        - summary: Executive summary
        - risk_score: Risk score (0-100)
        - clauses: List of critical clauses
    """
    # Combine chunks into a single text
    full_text = "\n\n".join(chunk["text"] for chunk in chunks)
    
    # Generate summary
    summary_prompt = _load_prompt("summarizer").format(text=full_text)
    summary = _call_deepseek(summary_prompt, model_name, temperature)
    
    # Extract clauses
    clause_prompt = _load_prompt("clause_extractor").format(text=full_text)
    clause_response = _call_deepseek(clause_prompt, model_name, temperature)
    
    try:
        clauses = json.loads(clause_response)
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        clauses = [{
            "clause_type": "Unknown",
            "risk_level": "Medium",
            "page": 1,
            "excerpt": clause_response,
            "justification": "Raw model output"
        }]
    
    # Calculate risk score
    risk_prompt = _load_prompt("risk_assessor").format(
        text=full_text,
        clauses=json.dumps(clauses, indent=2)
    )
    risk_response = _call_deepseek(risk_prompt, model_name, temperature)
    
    try:
        risk_data = json.loads(risk_response)
        risk_score = risk_data.get("risk_score", 50)  # Default to 50 if parsing fails
    except json.JSONDecodeError:
        risk_score = 50
    
    return {
        "summary": summary,
        "risk_score": risk_score,
        "clauses": clauses
    }
