"""
Analyzer module for processing NDA documents using DeepSeek.

Version v1: uses CSV for clause extraction and .replace() for prompt injection
to avoid any KeyError from str.format().
"""

from typing import List, Dict, Any
import os
import csv
import re
import requests
from pathlib import Path
import streamlit as st

# Load prompts directory
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

def _load_prompt(name: str) -> str:
    """Load a raw prompt template from disk."""
    return (PROMPTS_DIR / f"{name}.txt").read_text()

def _call_deepseek(
    prompt: str,
    model_name: str,
    temperature: float = 0.0
) -> str:
    """Call Fireworks DeepSeek API with deterministic output."""
    api_key = os.getenv("FIREWORKS_API_KEY") or (
        st.secrets["fireworks"]["api_key"] if "fireworks" in st.secrets else None
    )
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY not set in environment or Streamlit secrets.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": 2048,
        "top_p": 1,
        "top_k": 40
    }
    resp = requests.post("https://api.fireworks.ai/inference/v1/completions", headers=headers, json=data)
    if resp.status_code != 200:
        raise Exception(f"API call failed: {resp.status_code} - {resp.text}")
    return resp.json()["choices"][0]["text"]

def parse_clauses_csv(text: str) -> List[Dict[str, Any]]:
    """Parse CSV text into a list of clause dicts."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    clauses = []
    reader = csv.reader(lines)
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
    """
    Run analysis:
      1) Executive summary
      2) Clause extraction (CSV)
      3) Risk scoring via regex
    """
    # Combine text
    full_text = "\n\n".join(c["text"] for c in chunks[:top_k])

    # 1. Summary
    tmpl = _load_prompt("summarizer")
    summary_prompt = tmpl.replace("{text}", full_text)
    summary = _call_deepseek(summary_prompt, model_name, temperature).strip()

    # 2. Clause extraction CSV
    tmpl = _load_prompt("clause_extractor")
    clause_prompt = tmpl.replace("{text}", full_text)
    raw_csv = _call_deepseek(clause_prompt, model_name, temperature)
    with st.expander("🔍 Raw clauses CSV", expanded=False):
        st.code(raw_csv, language="text")

    clauses = parse_clauses_csv(raw_csv)
    if not clauses:
        st.warning("⚠️ No clauses extracted; inserting placeholder.")
        clauses = [{
            "clause_type": "Unknown",
            "risk_level": "Medium",
            "page": 1,
            "excerpt": "",
            "justification": "No valid lines parsed"
        }]

    # 3. Risk scoring
    tmpl = _load_prompt("risk_assessor")
    # Replace both placeholders safely
    risk_prompt = (
        tmpl
        .replace("{text}", full_text)
        .replace("{clauses_csv}", raw_csv)
    )
    raw_risk = _call_deepseek(risk_prompt, model_name, temperature)
    with st.expander("🔍 Raw risk output", expanded=False):
        st.code(raw_risk, language="text")

    m = re.search(r'\b(\d{1,3})\b', raw_risk)
    if m:
        risk_score = max(0, min(100, int(m.group(1))))
    else:
        st.warning("⚠️ Could not extract risk_score; defaulting to 50")
        risk_score = 50

    return {
        "summary": summary,
        "risk_score": risk_score,
        "clauses": clauses
    }
