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
    
    # Try to use different parameters based on whether we're expecting JSON
    json_expected = "json" in prompt.lower() or "array" in prompt.lower()
    
    data = {
        "model": model_name,
        "prompt": prompt,
        "temperature": 0.0 if json_expected else temperature,  # Force 0 temp for JSON
        "max_tokens": 2048,
        "top_p": 1,
        "top_k": 40
    }
    
    # If expecting JSON and API supports response format, specify it
    if json_expected:
        try:
            # Add response_format if we're requesting JSON
            data["response_format"] = {"type": "json_object"}
        except Exception:
            # If API doesn't support this parameter, continue without it
            pass

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
    
    Returns:
        A string containing a valid JSON array, or "[]" if extraction fails
    """
    # 1. First try direct parsing (best case scenario)
    try:
        text = text.strip()
        if text.startswith('[') and text.endswith(']'):
            json.loads(text)
            return text
    except json.JSONDecodeError:
        pass
    
    # 2. Try extracting from code blocks
    code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_match:
        try:
            extracted = code_match.group(1).strip()
            if extracted.startswith('[') and extracted.endswith(']'):
                json.loads(extracted)
                return extracted
        except json.JSONDecodeError:
            pass
    
    # 3. Try finding array pattern
    array_match = re.search(r"\[\s*{[\s\S]*}\s*\]", text)
    if array_match:
        try:
            extracted = array_match.group(0)
            json.loads(extracted)
            return extracted
        except json.JSONDecodeError:
            pass
    
    # 4. Aggressive cleanup - remove all formatting that could break JSON
    json_like = re.search(r"\[\s*{[\s\S]*}\s*\]", text)
    if json_like:
        extracted = json_like.group(0)
        
        # Remove all whitespace between tokens
        no_whitespace = re.sub(r'\s+', '', extracted)
        # Add minimal whitespace back for readability
        minimal_whitespace = no_whitespace.replace('{', '{ ').replace('}', ' }').replace(',', ', ')
        
        try:
            json.loads(minimal_whitespace)
            return minimal_whitespace
        except json.JSONDecodeError:
            pass
    
    # 5. LAST RESORT: Rebuild the JSON from regex-extracted clauses
    return rebuild_json_from_fragments(text)

def rebuild_json_from_fragments(text: str) -> str:
    """
    Last resort method to rebuild JSON by extracting key-value pairs.
    """
    # Look for properties in the format "key": "value" or "key":value
    clauses = []
    
    # Find anything that looks like the start of a JSON object
    object_matches = re.finditer(r'{(?:[^{}]|(?R))*}', text, re.DOTALL)
    
    for obj_match in object_matches:
        obj_text = obj_match.group(0)
        
        # Extract key-value pairs
        clause = {}
        
        # Extract clause_type
        type_match = re.search(r'"clause_type"\s*:\s*"([^"]*)"', obj_text)
        if type_match:
            clause["clause_type"] = type_match.group(1)
        else:
            continue  # Skip if no clause_type
        
        # Extract risk_level
        risk_match = re.search(r'"risk_level"\s*:\s*"([^"]*)"', obj_text)
        if risk_match:
            risk = risk_match.group(1)
            # Validate risk level
            if risk.lower() in ["high", "medium", "low"]:
                clause["risk_level"] = risk.capitalize()
            else:
                clause["risk_level"] = "Medium"  # Default
        else:
            clause["risk_level"] = "Medium"  # Default
        
        # Extract page
        page_match = re.search(r'"page"\s*:\s*(\d+)', obj_text)
        if page_match:
            try:
                clause["page"] = int(page_match.group(1))
            except ValueError:
                clause["page"] = 1
        else:
            clause["page"] = 1
        
        # Extract excerpt
        excerpt_match = re.search(r'"excerpt"\s*:\s*"([^"]*)"', obj_text)
        if excerpt_match:
            clause["excerpt"] = excerpt_match.group(1)
        else:
            clause["excerpt"] = "No excerpt available"
        
        # Extract justification
        just_match = re.search(r'"justification"\s*:\s*"([^"]*)"', obj_text)
        if just_match:
            clause["justification"] = just_match.group(1)
        else:
            clause["justification"] = "No justification provided"
        
        # Add complete clause
        if len(clause) == 5:  # Only add if we have all fields
            clauses.append(clause)
    
    # If no valid clauses found, try one more approach - look for fields directly
    if not clauses:
        # Extract all clause types
        types = re.findall(r'"clause_type"\s*:\s*"([^"]*)"', text)
        risks = re.findall(r'"risk_level"\s*:\s*"([^"]*)"', text)
        pages = re.findall(r'"page"\s*:\s*(\d+)', text)
        excerpts = re.findall(r'"excerpt"\s*:\s*"([^"]*)"', text)
        justs = re.findall(r'"justification"\s*:\s*"([^"]*)"', text)
        
        # Use the minimum length to avoid index errors
        min_len = min(len(types), len(risks), len(pages), len(excerpts), len(justs))
        
        for i in range(min_len):
            clause = {
                "clause_type": types[i],
                "risk_level": risks[i].capitalize() if risks[i].lower() in ["high", "medium", "low"] else "Medium",
                "page": int(pages[i]) if pages[i].isdigit() else 1,
                "excerpt": excerpts[i],
                "justification": justs[i]
            }
            clauses.append(clause)
    
    # If we found any clauses, convert back to JSON
    if clauses:
        return json.dumps(clauses)
    
    # If all else fails
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
    temperature: float = 0.0,  # Lowered to 0.0 for deterministic output
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
    clause_response = _call_deepseek(clause_prompt, model_name, temperature=0.0)  # Set to 0 for JSON

    # Debug view in UI
    with st.expander("🔍 Raw model output (clauses)", expanded=False):
        st.code(clause_response, language="json")

    try:
        # Try parsing directly first
        try:
            json_array = clause_response.strip()
            clauses = json.loads(json_array)
            st.success("✅ Direct JSON parsing successful")
        except json.JSONDecodeError:
            # If direct parsing fails, use extraction
            st.warning("⚠️ Using JSON extraction fallbacks")
            cleaned_json = extract_json_array(clause_response)
            clauses = json.loads(cleaned_json)
            
        # Additional validation
        if not isinstance(clauses, list):
            st.warning("⚠️ Model output not a list, wrapping")
            clauses = [clauses]
            
        # Ensure we have clauses
        if not clauses:
            st.warning("⚠️ No valid clauses found")
            clauses = [{
                "clause_type": "Unknown",
                "risk_level": "Medium",
                "page": 1,
                "excerpt": "Could not extract valid clauses",
                "justification": "The document needs manual review"
            }]
            
    except Exception as e:
        st.error(f"❌ Error processing clauses: {str(e)}")
        st.code(clause_response, language="text")
        
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
