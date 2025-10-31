# app/streamlit_app.py
import streamlit as st
import json
import sys
from pathlib import Path
import os

# Add the project root directory to Python path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from agents.graph import run_graph

st.set_page_config(page_title="Radiology → Scene-Graph (SGRRG style)", layout="wide")

st.title("Radiology Report → Scene-Graph (Chest ImaGenome / SGRRG style)")

st.markdown("""
Enter the radiology report text (findings + impression). The app will run multiple agents:
- sentence splitter
- rule-based candidate detector
- Gemini 2.5 Pro enricher
- Gemini-based verifier & normalizer
and return a scene-graph JSON using the 29 Chest ImaGenome anatomical objects.
""")

sample_path = Path(__file__).parent.parent / "examples" / "sample_report.txt"
default_text = ""
if sample_path.exists():
    default_text = sample_path.read_text()

report_text = st.text_area("Report text", value=default_text, height=300)

if st.button("Create scene-graph"):
    if not report_text.strip():
        st.warning("Paste a report first.")
    else:
        with st.spinner("Running agents (this calls Gemini 2.5 Pro)..."):
            try:
                sg = run_graph(report_text)
                st.success("Scene-graph created.")
                st.code(json.dumps(sg, indent=2), language="json")
                st.download_button("Download JSON", data=json.dumps(sg, indent=2), file_name="scene_graph.json", mime="application/json")
            except Exception as e:
                st.error(f"Error during agent run: {e}")
