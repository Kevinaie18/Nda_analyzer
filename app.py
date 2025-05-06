import os
from typing import Any, Dict, List
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from backend.loader import load_document
from backend.embedder import build_index, search
from backend.analyzer import run_full_analysis

load_dotenv()

# === Constants & Model Mapping ===
SUPPORTED_FORMATS = [".pdf", ".docx"]
AVAILABLE_MODELS = ["DeepSeek V3", "DeepSeek Reasoning"]
MODEL_MAP = {
    "DeepSeek V3": "accounts/fireworks/models/deepseek-v3",
    "DeepSeek Reasoning": "accounts/fireworks/models/deepseek-r1",
}
DEFAULT_MODEL = "DeepSeek V3"
DEFAULT_TOP_K = 8

def initialize_session_state():
    if "index" not in st.session_state: st.session_state.index = None
    if "docs_meta" not in st.session_state: st.session_state.docs_meta = {}
    if "analysis_results" not in st.session_state: st.session_state.analysis_results = {}

def display_usage_instructions():
    st.title("📄 NDA Analyzer")
    st.info("Upload un ou plusieurs NDA, sélectionne le modèle puis clique sur Run Analysis.")

def display_results(file_name: str):
    result = st.session_state.analysis_results[file_name]
    st.markdown(f"### Résultats pour `{file_name}`")
    
    # 1. Summary
    st.markdown("**Executive Summary**")
    st.write(result["summary"])
    
    # 2. Risk score
    st.metric("Risk Score", f"{result['risk_score']}/100")
    
    # 3. Clauses
    clauses = result["clauses"]
    clauses_df = pd.DataFrame(clauses)
    
    # Avertissement si placeholder
    if clauses and clauses[0]["clause_type"] == "Unknown":
        st.warning("⚠️ Extraction de clauses échouée, placeholder utilisé.")
    
    st.markdown("**Critical Clauses**")
    st.dataframe(
        clauses_df,
        column_config={
            "clause_type": "Clause Type",
            "risk_level": "Risk Level",
            "page": "Page",
            "excerpt": "Excerpt",
            "justification": "Justification"
        },
        hide_index=True
    )
    
    # 4. Export CSV des clauses
    csv_data = "\n".join(
        f'{c["clause_type"]},{c["risk_level"]},{c["page"]},"{c["excerpt"]}","{c["justification"]}"'
        for c in clauses
    )
    st.download_button(
        "📥 Download clauses CSV",
        csv_data,
        file_name=f"clauses_{file_name}.csv",
        mime="text/csv"
    )

def process_documents(uploaded_files, model_name, temperature, top_k):
    for up in uploaded_files:
        name = up.name
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            st.error(f"Unsupported format: {ext}")
            continue
        with st.spinner(f"Processing {name}…"):
            chunks = load_document(up)
            st.session_state.index, st.session_state.docs_meta = build_index(
                chunks, st.session_state.index, st.session_state.docs_meta, name
            )
            # map UI→API
            api_model = MODEL_MAP.get(model_name, MODEL_MAP[DEFAULT_MODEL])
            res = run_full_analysis(chunks, api_model, temperature, top_k)
            st.session_state.analysis_results[name] = res

def main():
    initialize_session_state()
    
    # API key
    api_key = st.secrets.get("fireworks", {}).get("api_key") or os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        st.error("FIREWORKS_API_KEY is not set. Edit your Streamlit secrets or .env.")
        st.stop()
    
    st.sidebar.title("Configuration")
    uploaded = st.sidebar.file_uploader("Upload NDA(s)", type=["pdf","docx"], accept_multiple_files=True)
    model = st.sidebar.selectbox("Modèle", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(DEFAULT_MODEL))
    temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    top_k = st.sidebar.number_input("Top K chunks", 1, 20, DEFAULT_TOP_K)
    if st.sidebar.button("Run Analysis") and uploaded:
        process_documents(uploaded, model, temp, top_k)
    
    if not uploaded:
        display_usage_instructions()
        return

    # Afficher résultats
    for fname in st.session_state.analysis_results:
        display_results(fname)

if __name__ == "__main__":
    main()
