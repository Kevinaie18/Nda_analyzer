"""
Analyzer module for processing NDA documents using DeepSeek.

Version v1.1: CSV‐only clause extraction with robust parsing and fallback,
plus reliable risk_score extraction.
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
        raise ValueError("FIREWORKS_API_KEY not set in env or secrets.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": 2048,
        "top_p": 1,
        "top_k": 40
    }
    resp = requests.post("https://api.fireworks.ai/inference/v1/completions",
                         headers=headers, json=data)
    if resp.status_code != 200:
        raise Exception(f"API call failed: {resp.status_code} - {resp.text}")
    return resp.json()["choices"][0]["text"]

def parse_clauses_csv(text: str) -> List[Dict[str, Any]]:
    """
    Parse CSV text into a list of clause dicts.
    Expected per-line format: clause_type,risk_level,page,excerpt,justification
    """
    # Remove code fences or unexpected wrappers
    text = re.sub(r"```[\s\S]*?```", "", text).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Skip header if present
    if lines and lines[0].lower().startswith("clause_type"):
        lines = lines[1:]
    clauses = []
    reader = csv.reader(lines)
    for parts in reader:
        # only accept lines with exactly 5 columns
        if len(parts) != 5:
            continue
        ctype, rlevel, page, excerpt, just = parts
        try:
            page_num = int(page)
        except ValueError:
            page_num = 1
        clauses.append({
            "clause_type": ctype.strip(),
            "risk_level": rlevel.strip(),
            "page": page_num,
            "excerpt": excerpt.strip(),
            "justification": just.strip()
        })
    return clauses

def run_full_analysis(
    chunks: List[Dict[str, Any]],
    model_name: str,
    temperature: float = 0.0,
    top_k: int = 8
) -> Dict[str, Any]:
    """
    Run a full analysis of an NDA document:
      1) Executive summary
      2) Clause extraction via CSV
      3) Risk scoring via regex
    """
    # 1. Combine top_k chunks
    full_text = "\n\n".join(chunk["text"] for chunk in chunks[:top_k])

    # 2. Executive summary
    tmpl = _load_prompt("summarizer")
    summary_prompt = tmpl.replace("{text}", full_text)
    summary = _call_deepseek(summary_prompt, model_name, temperature).strip()

    # 3. Clause extraction (CSV)
    tmpl = _load_prompt("clause_extractor")
    clause_prompt = tmpl.replace("{text}", full_text)
    raw_csv = _call_deepseek(clause_prompt, model_name, temperature)
    with st.expander("🔍 Raw clauses CSV", expanded=True):
        st.code(raw_csv, language="text")

    clauses = parse_clauses_csv(raw_csv)
    if not clauses:
        st.warning("⚠️ No clauses extracted via CSV; retrying with relaxed parsing…")
        # fallback: accept lines with >=3 columns
        loose_clauses = []
        for line in [l for l in raw_csv.splitlines() if l.strip()]:
            parts = [p.strip() for p in re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', line)]
            if len(parts) >= 3:
                loose_clauses.append({
                    "clause_type": parts[0],
                    "risk_level": parts[1] if len(parts) > 1 else "Medium",
                    "page": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1,
                    "excerpt": parts[3] if len(parts) > 3 else "",
                    "justification": parts[4] if len(parts) > 4 else ""
                })
        if loose_clauses:
            clauses = loose_clauses
        else:
            # hard placeholder
            clauses = [{
                "clause_type": "Unknown",
                "risk_level": "Medium",
                "page": 1,
                "excerpt": "",
                "justification": "Unable to extract clauses"
            }]

    # 4. Risk scoring
    tmpl = _load_prompt("risk_assessor")
    # clauses_csv: rebuild CSV from parsed clauses
    csv_lines = "\n".join(
        f'{c["clause_type"]},{c["risk_level"]},{c["page"]},"{c["excerpt"]}","{c["justification"]}"'
        for c in clauses
    )
    risk_prompt = tmpl.replace("{text}", full_text).replace("{clauses_csv}", csv_lines)
    raw_risk = _call_deepseek(risk_prompt, model_name, temperature)
    with st.expander("🔍 Raw risk output", expanded=True):
        st.code(raw_risk, language="text")

    m = re.search(r'\b(\d{1,3})\b', raw_risk)
    risk_score = int(m.group(1)) if m else 50

    return {
        "summary": summary,
        "risk_score": max(0, min(100, risk_score)),
        "clauses": clauses
    }
