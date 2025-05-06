"""
Analyzer module for processing NDA documents using DeepSeek.

This module provides functionality to analyze NDA documents using the DeepSeek
language model, extracting summaries, risk scores, and critical clauses.
"""

from typing import List, Dict, Any, Optional
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
        # Add JSON formatting wrapper
        base_prompt += """

RESPONSE FORMAT REQUIREMENTS:
You must respond ONLY with a valid JSON array of clause objects with NO additional text.
Do not include markdown formatting, explanation text, or code blocks.

JSON SCHEMA:
[
  {
    "clause_type": string,          // Type of legal clause
    "risk_level": "High" | "Medium" | "Low",  // Risk assessment
    "page": number,                 // Page number where clause appears
    "excerpt": string,              // Extract of max 40 words
    "justification": string         // Why this clause has the assigned risk level
  },
  // Additional clause objects as needed
]

EXAMPLE RESPONSE:
[
  {
    "clause_type": "Non-compete",
    "risk_level": "High",
    "page": 2,
    "excerpt": "Party shall not engage in similar business for 5 years globally",
    "justification": "Duration and scope are overly restrictive"
  }
]

Return ONLY the JSON array.
"""
    
    return base_prompt

def _call_deepseek(
    prompt: str,
    model_name: str,
    temperature: float = 0.2  # Lowered temperature for more consistent structured output
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

    data = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": 2048,
        "top_p": 1,
        "top_k": 40,
        "response_format": {"type": "json_object"}  # Request JSON format if API supports it
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
    """
    Extract valid JSON array from model output with robust handling of various formats.
    
    Returns:
        A string containing a valid JSON array, or "[]" if extraction fails
    """
    # First try direct JSON parsing (maybe it's already valid)
    try:
        # If the text is already valid JSON array, return it
        if text.strip().startswith('[') and text.strip().endswith(']'):
            json.loads(text)
            return text.strip()
    except json.JSONDecodeError:
        pass
    
    # Extract from code blocks (common in markdown)
    code_block_pattern = r"```(?:json)?\s*(\[\s*\{.*?\}\s*\])\s*```"
    code_match = re.search(code_block_pattern, text, re.DOTALL)
    if code_match:
        try:
            # Validate it's actually JSON
            json_str = code_match.group(1)
            json.loads(json_str)  # Test if valid
            return json_str
        except json.JSONDecodeError:
            pass
    
    # Try to find array pattern with multiple objects - greedy approach
    array_pattern = r"\[\s*\{[^][]*(?:\][^][]*\[[^][]*)*\}\s*\]"
    array_match = re.search(array_pattern, text, re.DOTALL)
    if array_match:
        try:
            json_str = array_match.group(0)
            json.loads(json_str)  # Test if valid
            return json_str
        except json.JSONDecodeError:
            pass
    
    # Try progressive cleaning approach for malformed JSON
    json_like = re.search(r"(\[\s*\{.*\}\s*\])", text, re.DOTALL)
    if json_like:
        potential_json = json_like.group(1)
        
        # Fix common JSON issues
        fixes = [
            (r',\s*\]', ']'),                  # Remove trailing commas
            (r'(["\w])\s+([{\["]))', r'\1, \2'),  # Add missing commas
            (r'\\n\s*"', ' "'),                # Fix newline issues
            (r'"\s*:\s*"([^"]*?)"([,}])', r'": "\1"\2'),  # Fix quote nesting
        ]
        
        for pattern, replacement in fixes:
            potential_json = re.sub(pattern, replacement, potential_json)
            
        try:
            json.loads(potential_json)  # Test if valid
            return potential_json
        except json.JSONDecodeError:
            pass
    
    # If all attempts fail, return empty array
    return "[]"

def parse_clauses(json_text: str) -> List[Dict[str, Any]]:
    """
    Parse and validate clause data from JSON text, ensuring all fields are present.
    
    Returns:
        List of validated clause objects
    """
    try:
        clauses = json.loads(json_text)
        validated_clauses = []
        
        # Ensure we have a list
        if isinstance(clauses, dict) and "clauses" in clauses:
            clauses = clauses["clauses"]
        elif not isinstance(clauses, list):
            return []
            
        # Validate each clause has required fields
        required_fields = {"clause_type", "risk_level", "page", "excerpt", "justification"}
        risk_levels = {"High", "Medium", "Low"}
        
        for clause in clauses:
            # Ensure all fields exist
            if not all(field in clause for field in required_fields):
                # Add missing fields with default values
                for field in required_fields:
                    if field not in clause:
                        if field == "risk_level":
                            clause[field] = "Medium"
                        elif field == "page":
                            clause[field] = 1
                        else:
                            clause[field] = "Not specified"
            
            # Validate risk level
            if clause["risk_level"] not in risk_levels:
                clause["risk_level"] = "Medium"  # Default to medium
                
            # Ensure page is an integer
            try:
                clause["page"] = int(clause["page"])
            except (ValueError, TypeError):
                clause["page"] = 1
                
            validated_clauses.append(clause)
            
        return validated_clauses
    except json.JSONDecodeError:
        st.error("Failed to parse JSON data")
        return []

def run_full_analysis(
    chunks: List[Dict[str, Any]],
    model_name: str,
    temperature: float = 0.2,  # Lower temperature for structured outputs
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
    # Combine top_k chunks into one string
    full_text = "\n\n".join(chunk["text"] for chunk in chunks[:top_k])

    # 1. Generate executive summary
    summary_prompt = _load_prompt("summarizer").format(text=full_text)
    summary = _call_deepseek(summary_prompt, model_name, temperature)

    # 2. Extract critical clauses
    clause_prompt = _load_prompt("clause_extractor").format(text=full_text)
    clause_response = _call_deepseek(clause_prompt, model_name, temperature=0.1)  # Even lower temp for JSON

    # Debug view in UI
    with st.expander("🔍 Raw model output (clauses)", expanded=False):
        st.code(clause_response, language="json")

    try:
        cleaned_json = extract_json_array(clause_response)
        clauses = parse_clauses(cleaned_json)
        
        if not clauses:
            st.warning("⚠️ Unable to extract valid clauses from model output.")
            clauses = [{
                "clause_type": "Unknown",
                "risk_level": "Medium",
                "page": 1,
                "excerpt": clause_response.strip()[:300] if clause_response else "No content",
                "justification": "Model output not JSON-parsable"
            }]
    except Exception as e:
        st.warning(f"⚠️ Error processing clauses: {str(e)}")
        clauses = [{
            "clause_type": "Error",
            "risk_level": "Medium",
            "page": 1,
            "excerpt": clause_response[:300] if clause_response else "No content",
            "justification": f"Processing error: {str(e)}"
        }]

    # 3. Assess risk score
    risk_prompt = _load_prompt("risk_assessor").format(
        text=full_text,
        clauses=json.dumps(clauses, indent=2)
    )
    risk_response = _call_deepseek(risk_prompt, model_name, temperature)

    try:
        # Try to extract risk score as JSON
        risk_pattern = r'\{\s*"risk_score"\s*:\s*(\d+)\s*\}'
        risk_match = re.search(risk_pattern, risk_response)
        
        if risk_match:
            risk_score = int(risk_match.group(1))
        else:
            # Try parsing complete response as JSON
            risk_data = json.loads(risk_response)
            risk_score = risk_data.get("risk_score", 50)
    except (json.JSONDecodeError, ValueError):
        risk_score = 50  # Default risk score

    return {
        "summary": summary.strip(),
        "risk_score": risk_score,
        "clauses": clauses
    }
