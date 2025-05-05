"""
NDA Analyzer - Streamlit Application

This module provides a web interface for analyzing Non-Disclosure Agreements (NDAs)
using AI-powered document analysis. It integrates with various backend services
for document processing, embedding generation, and AI analysis.
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import backend modules
from backend.loader import load_document
from backend.embedder import build_index, search
from backend.analyzer import run_full_analysis

# Constants
SUPPORTED_FORMATS = [".pdf", ".docx"]

AVAILABLE_MODELS = [
    "DeepSeek V3",
    "DeepSeek Reasoning"
]

MODEL_MAP = {
    "DeepSeek V3": "accounts/fireworks/models/deepseek-v3",
    "DeepSeek Reasoning": "accounts/fireworks/models/deepseek-r1"
}

DEFAULT_MODEL = "DeepSeek V3"
DEFAULT_TOP_K = 8

def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "index" not in st.session_state:
        st.session_state.index = None
    if "docs_meta" not in st.session_state:
        st.session_state.docs_meta = {}
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}

def display_usage_instructions() -> None:
    """Display initial usage instructions."""
    st.title("📄 NDA Analyzer")
    st.markdown("""
    ### Welcome to NDA Analyzer!
    
    This tool helps you analyze Non-Disclosure Agreements using AI. Here's how to use it:
    
    1. **Upload Documents**: Use the sidebar to upload one or more NDA documents (PDF or DOCX)
    2. **Configure Analysis**: Adjust the analysis parameters in the sidebar
    3. **Run Analysis**: Click the "Run Analysis" button to process your documents
    4. **Review Results**: View the summary, risk score, and extracted clauses
    5. **Search**: Use the semantic search feature to find specific information
    
    Get started by uploading your documents in the sidebar!
    """)

def process_documents(
    uploaded_files: List[Any],
    model_name: str,
    temperature: float,
    top_k: int
) -> None:
    """Process uploaded documents and store results in session state."""
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_ext = Path(file_name).suffix.lower()
        
        if file_ext not in SUPPORTED_FORMATS:
            st.error(f"Unsupported file format: {file_ext}")
            continue
            
        try:
            with st.spinner(f"Processing {file_name}..."):
                # Load and chunk document
                chunks = load_document(uploaded_file)
                
                # Build or update FAISS index
                st.session_state.index, st.session_state.docs_meta = build_index(
                    chunks,
                    st.session_state.index,
                    st.session_state.docs_meta,
                    file_name
                )
                
                # Map model name to API-compatible name
                model_api_name = MODEL_MAP.get(model_name, "accounts/fireworks/models/deepseek-v3")

                # Run analysis
                analysis_result = run_full_analysis(
                    chunks,
                    model_name=model_api_name,
                    temperature=temperature,
                    top_k=top_k
                )
                
                st.session_state.analysis_results[file_name] = analysis_result
                
        except Exception as e:
            error_msg = str(e)
            st.error(f"Error processing {file_name}: {error_msg}")
            
            # Provide helpful guidance based on error type
            if "Model not found" in error_msg:
                st.info("""
                The selected model is not available. Please try:
                1. Using a different model from the dropdown
                2. Checking your Fireworks API key permissions
                """)
            elif "API key" in error_msg.lower():
                st.info("""
                There's an issue with your API key. Please:
                1. Check if your FIREWORKS_API_KEY is set in .env or secrets.toml
                2. Verify the key is valid in your Fireworks dashboard
                """)
            elif "rate limit" in error_msg.lower():
                st.info("""
                You've hit the API rate limit. Please:
                1. Wait a few minutes before trying again
                2. Check your usage in the Fireworks dashboard
                """)

def display_results(file_name: str) -> None:
    """Display analysis results for a specific file."""
    result = st.session_state.analysis_results[file_name]
    
    # Display summary
    st.markdown("### Executive Summary")
    st.markdown(result["summary"])
    
    # Display risk score
    st.metric("Risk Score", f"{result['risk_score']:.1f}/100")
    
    # Display clauses
    st.markdown("### Critical Clauses")
    clauses_df = pd.DataFrame(result["clauses"])
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

def main() -> None:
    """Main application entry point."""
    initialize_session_state()
    
    # Check for API key from secrets or fallback to .env
    api_key = st.secrets["fireworks"]["api_key"] if "fireworks" in st.secrets else os.getenv("FIREWORKS_API_KEY")

    if not api_key:
        st.error("""
        FIREWORKS_API_KEY is not set.

        Please either:
        1. Add it to `.streamlit/secrets.toml` under [fireworks]
        2. Or set it locally in a .env file at the project root

        Then restart the application.
        """)
        st.stop()
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload NDA Documents",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    
    model_name = st.sidebar.selectbox(
        "Model Name",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
        help="Select the AI model to use for analysis."
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make the output more creative but less focused"
    )
    
    top_k = st.sidebar.number_input(
        "Top K for Retrieval",
        min_value=1,
        max_value=20,
        value=DEFAULT_TOP_K,
        help="Number of most relevant chunks to consider for analysis"
    )
    
    run_button = st.sidebar.button("Run Analysis")
    
    # Main content area
    if not uploaded_files:
        display_usage_instructions()
        return
    
    if run_button:
        process_documents(uploaded_files, model_name, temperature, top_k)
    
    # Display results for each processed file
    for file_name in st.session_state.analysis_results:
        st.markdown(f"## {file_name}")
        display_results(file_name)
        
        # Semantic search
        st.markdown("### Semantic Search")
        search_query = st.text_input(
            "Search within document",
            key=f"search_{file_name}"
        )
        
        if search_query:
            try:
                search_results = search(
                    st.session_state.index,
                    search_query,
                    st.session_state.docs_meta,
                    file_name,
                    top_k=top_k
                )
                for result in search_results:
                    st.markdown(f"**Page {result['page']}**: {result['excerpt']}")
            except Exception as e:
                st.error(f"Search error: {str(e)}")
        
        # Export button
        export_md = f"""# NDA Analysis Report: {file_name}

## Executive Summary
{st.session_state.analysis_results[file_name]['summary']}

## Risk Score
{st.session_state.analysis_results[file_name]['risk_score']}/100

## Critical Clauses
{clauses_df.to_markdown(index=False)}
"""
        
        st.download_button(
            "Download Report",
            export_md,
            file_name=f"nda_analysis_{file_name}.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()
