"""
Analyzer module for processing NDA documents using DeepSeek.

This version uses CSV‐based clause extraction and robust risk_score parsing
to avoid any JSON parsing errors.
"""

from typing import List, Dict, Any
import os
import csv
import re
import requests
from pathlib import Path
import streamlit as st

# Load prompts
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

def _load_prompt(name: str) -> str:
    """Load a prompt from the prompts directory."""
    return (PROMPTS_DIR / f"{name}.txt").read_text().strip()

def _call_deepseek(
    prompt: str,
    model_name: str,
    temperature: float = 0.0
) -> str:
    """Call DeepSeek via Fireworks API, forcing deterministic output for structured data."""
    api_key = os.getenv("FIREWORKS_API_KEY") or (
        st.secrets["fireworks"]["api_key"] if "fireworks" in st.secrets else None
    )
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY is not set in environment or Streamlit secrets.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": 2048,
        "top_p": 1,
        "top_k": 40
    }
    # hint for JSON/CSV? always force deterministic
    response = requests.post(
        "https://api.fireworks.ai/inference/v1/completions",
        headers=headers,
        json=data
    )
    if response.status_code != 200:
        raise Exception(f"API call failed: {response.status_code} - {response.text}")
    return response.json()["choices"][0]["text"]

def parse_clauses_csv(text: str) -> List[Dict[str, Any]]:
    """
    Parse CSV text into a list of clause dicts.
    Expected format per line: clause_type,risk_level,page,excerpt,justification
    """
    lines = [l for l in text.strip().splitlines() if l.strip()]
    reader = csv.reader(lines)
    clauses = []
    for parts in reader:
        if len(parts) != 5:
            continue
        ctype, rlevel, page, excerpt, just = parts
        try:
            page_num = int(page)
        except ValueError:
            page_num = 1
        clauses.append({
            "clause_type": ctype,
            "risk_level": rlevel,
            "page": page_num,
            "excerpt": excerpt,
            "justification": just
        })
    return clauses

def run_full_analysis(
    chunks: List[Dict[str, Any]],
    model_name: str,
    temperature: float = 0.0,
    top_k: int = 8
) -> Dict[str, Any]:
    """Run summary, CSV clause extraction, and robust risk scoring."""
    # 1. Build full text context
    full_text = "\n\n".join(chunk["text"] for chunk in chunks[:top_k])

    # 2. Executive summary
    summary_prompt = _load_prompt("summarizer").format(text=full_text)
    summary = _call_deepseek(summary_prompt, model_name, temperature)

    # 3. Clause extraction (CSV)
    clause_prompt = _load_prompt("clause_extractor").format(text=full_text)
    raw_csv = _call_deepseek(clause_prompt, model_name, temperature)
    with st.expander("🔍 Raw clauses CSV", expanded=False):
        st.code(raw_csv, language="text")

    clauses = parse_clauses_csv(raw_csv)
    if not clauses:
        st.warning("⚠️ No clauses extracted via CSV; using placeholder.")
        clauses = [{
            "clause_type": "Unknown",
            "risk_level": "Medium",
            "page": 1,
            "excerpt": "",
            "justification": "No valid CSV lines parsed"
        }]

    # 4. Risk scoring
    # use a prompt that returns plain JSON or plain number, then extract via regex
    risk_prompt = _load_prompt("risk_assessor").format(
        text=full_text,
        clauses=[c for c in clauses]
    )
    risk_resp = _call_deepseek(risk_prompt, model_name, temperature)
    with st.expander("🔍 Raw risk output", expanded=False):
        st.code(risk_resp, language="text")

    # robustly extract first integer encountered as risk_score
    trimmed = risk_resp.strip()
    match = re.search(r'\b(\d{1,3})\b', trimmed)
    if match:
        risk_score = int(match.group(1))
    else:
        st.warning("⚠️ Could not extract numeric risk_score; defaulting to 50")
        risk_score = 50

    return {
        "summary": summary.strip(),
        "risk_score": risk_score,
        "clauses": clauses
    }
