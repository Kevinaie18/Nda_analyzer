"""
Analyzer module for processing NDA documents using DeepSeek.

This module provides functionality to analyze NDA documents using the DeepSeek
language model, extracting summaries, risk scores, and critical clauses.
"""

from typing import List, Dict, Any
import os
import json
import re
import requests
from pathlib import Path
import streamlit as st  # required for secrets and UI debug

# Load prompts
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

def _load_prompt(name: str) -> str:
    """Load a prompt from the prompts directory."""
    prompt_path = PROMPTS_DIR / f"{name}.txt"
    with open(prompt_path, "r") as f:
        return f.read().strip()

def _call_deepseek(
    prompt: str,
    model_name: str,
    temperature: float = 0.7
) -> str:
    """
    Call the DeepSeek model via Fireworks API.

    Args:
        prompt: The input prompt
        model_name: Full model ID (e.g. 'accounts/fireworks/models/deepseek-v3')
        temperature: Sampling temperature

    Returns:
        The model's raw text response
    """
    api_key = os.getenv("FIREWORKS_API_KEY") or \
              (st.secrets["fireworks"]["api_key"] if "fireworks" in st.secrets else None)

    if not api_key:
        raise ValueError("FIREWORKS_API_KEY is not set in environment or secrets.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": 2048,
        "top_p": 1,
        "top_k": 40
    }

    response = requests.post(
        "https://api.fireworks.ai/inference/v1/completions",
        headers=headers,
        json=data
    )

    if response.status_code != 200:
        raise Exception(f"API call failed: {response.status_code} - {response.text}")

    return response.json()["choices"][0]["text"]

def extract_json_array(text: str) -> str:
    """Extract the first JSON array found in a string."""
    match = re.search(r"\[\s*{.*?}\s*\]", text, re.DOTALL)
    return match.group(0) if match else "[]"

def run_full_analysis(
    chunks: List[Dict[str, Any]],
    model_name: str,
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
        Dictionary with keys: summary, risk_score, clauses
    """
    full_text = "\n\n".join(chunk["text"] for chunk in chunks[:top_k])

    # 1. Generate summary
    summary_prompt = _load_prompt("summarizer").format(text=full_text)
    summary = _call_deepseek(summary_prompt, model_name, temperature)

    # 2. Extract clauses
    clause_prompt = _load_prompt("clause_extractor").format(text=full_text)
    clause_response = _call_deepseek(clause_prompt, model_name, temperature)

    try:
        cleaned_json = extract_json_array(clause_response)
        clauses = json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        st.warning("⚠️ Failed to parse clause list. Showing raw output.")

        # Debug info in logs + interface
        print("❌ JSONDecodeError:", e)
        print("↪️ Raw clause_response:", clause_response[:500])
        st.expander("🔍 Raw clause response").code(clause_response)

        clauses = [{
            "clause_type": "Unknown",
            "risk_level": "Medium",
            "page": 1,
            "excerpt": clause_response.strip()[:300],
            "justification": "Raw model output (unparsed)"
        }]

    # 3. Assess risk score
    risk_prompt = _load_prompt("risk_assessor").format(
        text=full_text,
        clauses=json.dumps(clauses, indent=2)
    )
    risk_response = _call_deepseek(risk_prompt, model_name, temperature)

    try:
        risk_data = json.loads(risk_response)
        risk_score = risk_data.get("risk_score", 50)
    except json.JSONDecodeError:
        risk_score = 50

    return {
        "summary": summary.strip(),
        "risk_score": risk_score,
        "clauses": clauses
    }
