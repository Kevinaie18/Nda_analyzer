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
import streamlit as st

# Load prompts
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

def _load_prompt(name: str) -> str:
    """Load a prompt from the prompts directory."""
    base_prompt = (PROMPTS_DIR / f"{name}.txt").read_text().strip()
    
    if name == "clause_extractor":
        # Simplified prompt with absolute minimum formatting requirements
        base_prompt += """

OUTPUT FORMAT:
Return a JSON array of objects ONLY. No explanations or other text.
Each object must have these exact keys: "clause_type", "risk_level", "page", "excerpt", "justification"
Risk level must be exactly one of: "High", "Medium", "Low"
Page must be a number.

MINIMAL FORMAT EXAMPLE:
[{"clause_type":"Non-compete","risk_level":"High","page":2,"excerpt":"text","justification":"reason"}]

DO NOT use pretty printing, line breaks, or indentation.
DO NOT add comments or explanations.
DO NOT wrap the output in code blocks.
DO NOT use any text before or after the JSON array.
DO NOT use any formatting that isn't required for valid JSON.
"""
    
    return base_prompt

def _call_deepseek(
    prompt: str,
    model_name: str,
    temperature: float = 0.0  # Deterministic output for structured data
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
    api_key = os.getenv("FIREWORKS_API_KEY") or (
        st.secrets["fireworks"]["api_key"] if "fireworks" in st.secrets else None
    )

    if not api_key:
        raise ValueError("FIREWORKS_API_KEY is not set in environment or Streamlit secrets.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Determine if we expect JSON
    json_expected = "json" in prompt.lower() or "array" in prompt.lower()
    
    data = {
        "model": model_name,
        "prompt": prompt,
        "temperature": 0.0 if json_expected else temperature,
        "max_tokens": 2048,
        "top_p": 1,
        "top_k": 40
    }
    
    # If API supports a JSON response hint, include it
    if json_expected:
        data["response_format"] = {"type": "json_object"}
    
    response = requests.post(
        "https://api.fireworks.ai/inference/v1/completions",
        headers=headers,
        json=data
    )

    if response.status_code != 200:
        raise Exception(f"API call failed: {response.status_code} - {response.text}")

    return response.json()["choices"][0]["text"]

def extract_json_array(text: str) -> str:
    """
    Extract valid JSON array from model output with extremely robust handling.
    Prioritizes recovering valid JSON over preserving exact formatting.
    """
    # 1. Direct valid JSON?
    t = text.strip()
    if t.startswith('[') and t.endswith(']'):
        try:
            json.loads(t)
            return t
        except json.JSONDecodeError:
            pass

    # 2. Inside code fences?
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        cand = m.group(1).strip()
        if cand.startswith('[') and cand.endswith(']'):
            try:
                json.loads(cand)
                return cand
            except json.JSONDecodeError:
                pass

    # 3. Simple array pattern
    m = re.search(r"\[\s*{[\s\S]*}\s*\]", text)
    if m:
        cand = m.group(0)
        try:
            json.loads(cand)
            return cand
        except json.JSONDecodeError:
            pass

    # 4. Aggressive cleanup
    if m:
        extracted = m.group(0)
        no_ws = re.sub(r'\s+', '', extracted)
        minimal = no_ws.replace('{', '{ ').replace('}', ' }').replace(',', ', ')
        try:
            json.loads(minimal)
            return minimal
        except json.JSONDecodeError:
            pass

    # 5. Last resort: rebuild from fragments
    return rebuild_json_from_fragments(text)

def rebuild_json_from_fragments(text: str) -> str:
    """
    Last resort method to rebuild JSON by extracting key-value pairs.
    """
    clauses = []
    # Find JSON-like object blocks
    for obj in re.finditer(r'{(?:[^{}]|(?R))*}', text, re.DOTALL):
        o = obj.group(0)
        c = {}
        m = re.search(r'"clause_type"\s*:\s*"([^"]*)"', o)
        if not m:
            continue
        c["clause_type"] = m.group(1)
        m = re.search(r'"risk_level"\s*:\s*"([^"]*)"', o)
        rl = m.group(1) if m else "Medium"
        c["risk_level"] = rl.capitalize() if rl.lower() in ["high","medium","low"] else "Medium"
        m = re.search(r'"page"\s*:\s*(\d+)', o)
        c["page"] = int(m.group(1)) if m else 1
        m = re.search(r'"excerpt"\s*:\s*"([^"]*)"', o)
        c["excerpt"] = m.group(1) if m else "No excerpt"
        m = re.search(r'"justification"\s*:\s*"([^"]*)"', o)
        c["justification"] = m.group(1) if m else "No justification"
        clauses.append(c)

    # If none, try parallel lists
    if not clauses:
        types = re.findall(r'"clause_type"\s*:\s*"([^"]*)"', text)
        risks = re.findall(r'"risk_level"\s*:\s*"([^"]*)"', text)
        pages = re.findall(r'"page"\s*:\s*(\d+)', text)
        exs = re.findall(r'"excerpt"\s*:\s*"([^"]*)"', text)
        js = re.findall(r'"justification"\s*:\s*"([^"]*)"', text)
        n = min(len(types), len(risks), len(pages), len(exs), len(js))
        for i in range(n):
            clauses.append({
                "clause_type": types[i],
                "risk_level": risks[i].capitalize() if risks[i].lower() in ["high","medium","low"] else "Medium",
                "page": int(pages[i]),
                "excerpt": exs[i],
                "justification": js[i]
            })

    return json.dumps(clauses) if clauses else "[]"

def parse_clauses(json_text: str) -> List[Dict[str, Any]]:
    """
    Parse and validate clause data from JSON text, ensuring all fields are present.
    """
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        st.error("Failed to parse JSON data")
        return []

    # Unwrap if wrapped
    if isinstance(data, dict) and "clauses" in data:
        data = data["clauses"]
    if not isinstance(data, list):
        return []

    valid = []
    for c in data:
        # Ensure required keys
        for key, default in [
            ("clause_type", "Unknown"),
            ("risk_level", "Medium"),
            ("page", 1),
            ("excerpt", ""),
            ("justification", "")
        ]:
            if key not in c:
                c[key] = default
        # Normalize types
        if c["risk_level"] not in ["High","Medium","Low"]:
            c["risk_level"] = "Medium"
        try:
            c["page"] = int(c["page"])
        except Exception:
            c["page"] = 1
        valid.append(c)
    return valid

def run_full_analysis(
    chunks: List[Dict[str, Any]],
    model_name: str,
    temperature: float = 0.0,  # Deterministic for JSON
    top_k: int = 8
) -> Dict[str, Any]:
    """
    Run a full analysis of an NDA document.
    """
    full_text = "\n\n".join(chunk["text"] for chunk in chunks[:top_k])

    # 1. Summary
    summary_prompt = _load_prompt("summarizer").format(text=full_text)
    summary = _call_deepseek(summary_prompt, model_name, temperature)

    # 2. Clauses
    clause_prompt = _load_prompt("clause_extractor").format(text=full_text)
    clause_response = _call_deepseek(clause_prompt, model_name, temperature)

    # Show raw for debug
    with st.expander("🔍 Raw clauses output", expanded=False):
        st.code(clause_response, language="text")

    # Parse clauses
    try:
        # Try direct parse
        clauses = json.loads(clause_response.strip())
        st.success("✅ Direct JSON parse succeeded")
    except Exception:
        st.warning("⚠️ Direct parse failed; using extraction")
        json_array = extract_json_array(clause_response)
        clauses = json.loads(json_array)

    clauses = parse_clauses(json.dumps(clauses))

    if not clauses:
        st.warning("⚠️ No clauses extracted; inserting placeholder")
        clauses = [{
            "clause_type": "Unknown",
            "risk_level": "Medium",
            "page": 1,
            "excerpt": "",
            "justification": "No clauses could be parsed"
        }]

    # 3. Risk score
    risk_prompt = _load_prompt("risk_assessor").format(
        text=full_text,
        clauses=json.dumps(clauses, indent=2)
    )
    risk_resp = _call_deepseek(risk_prompt, model_name, temperature)

    # Extract numeric risk_score
    risk_match = re.search(r'"risk_score"\s*:\s*(\d+)', risk_resp)
    if risk_match:
        risk_score = int(risk_match.group(1))
    else:
        try:
            risk_score = json.loads(risk_resp).get("risk_score", 50)
        except Exception:
            risk_score = 50

    return {
        "summary": summary.strip(),
        "risk_score": risk_score,
        "clauses": clauses
    }
