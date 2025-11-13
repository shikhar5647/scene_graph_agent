# app/streamlit_app.py
import streamlit as st
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add the project root directory to Python path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from agents.graph import run_graph

st.set_page_config(page_title="Radiology → Scene-Graph Matrix", layout="wide")

st.title("Radiology Report → Scene-Graph Matrix (SGRRG style)")

st.markdown("""
Enter chest X-ray radiology report text. The app will extract findings and generate a 
**29 objects × attributes matrix** where:
- **+1** = attribute present
- **0** = attribute absent/not mentioned  
- **-1** = attribute uncertain

Pipeline: sentence splitting → candidate detection → Gemini enrichment → verification → matrix building
""")

# Load sample report
sample_path = Path(__file__).parent.parent / "examples" / "sample_report.txt"
default_text = ""
if sample_path.exists():
    default_text = sample_path.read_text()

report_text = st.text_area("Report text", value=default_text, height=300)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Generate Scene-Graph Matrix", type="primary"):
        if not report_text.strip():
            st.warning("Please paste a report first.")
        else:
            with st.spinner("Running pipeline (calling Gemini API)..."):
                try:
                    result = run_graph(report_text)
                    matrix = result["matrix"]
                    metadata = result["metadata"]
                    
                    # Store in session state
                    st.session_state['matrix'] = matrix
                    st.session_state['metadata'] = metadata
                    
                    st.success("✓ Scene-graph matrix generated!")
                    
                except Exception as e:
                    st.error(f"Error during pipeline execution: {e}")
                    import traceback
                    st.code(traceback.format_exc())

with col2:
    if st.button("Clear Results"):
        if 'matrix' in st.session_state:
            del st.session_state['matrix']
        if 'metadata' in st.session_state:
            del st.session_state['metadata']
        st.rerun()

# Display results if available
if 'matrix' in st.session_state and 'metadata' in st.session_state:
    matrix = st.session_state['matrix']
    metadata = st.session_state['metadata']
    
    st.divider()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Matrix View", "🔍 Findings Summary", "📥 Export", "ℹ️ Info"])
    
    with tab1:
        st.subheader("Objects × Attributes Matrix")
        
        # Create DataFrame for display
        df = pd.DataFrame(
            matrix,
            index=metadata['objects'],
            columns=metadata['attributes']
        )
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            show_only_positive = st.checkbox("Show only positive findings (+1)", value=False)
        with col2:
            show_only_mentioned = st.checkbox("Show only mentioned objects", value=True)
        
        if show_only_mentioned:
            # Filter rows where at least one attribute is non-zero
            mask = (matrix != 0).any(axis=1)
            df_display = df[mask]
        else:
            df_display = df
        
        if show_only_positive:
            # Show only columns with at least one positive value
            mask = (df_display == 1).any(axis=0)
            df_display = df_display.loc[:, mask]
        
        # Color mapping for display
        def color_matrix(val):
            if val == 1:
                return 'background-color: #d4edda; color: #155724'  # green
            elif val == -1:
                return 'background-color: #fff3cd; color: #856404'  # yellow
            elif val == 0:
                return 'background-color: #f8f9fa; color: #6c757d'  # gray
            return ''
        
        st.dataframe(
            df_display.style.applymap(color_matrix),
            use_container_width=True,
            height=600
        )
        
        st.caption(f"Matrix shape: {df_display.shape[0]} objects × {df_display.shape[1]} attributes")
    
    with tab2:
        st.subheader("Extracted Findings Summary")
        
        findings = metadata.get('findings_summary', {})
        
        if findings:
            for obj_name, attrs in findings.items():
                if attrs:  # Only show objects with findings
                    st.markdown(f"**{obj_name}**")
                    
                    positive = [k for k, v in attrs.items() if v == 1]
                    negative = [k for k, v in attrs.items() if v == 0]
                    uncertain = [k for k, v in attrs.items() if v == -1]
                    
                    if positive:
                        st.markdown(f"- ✓ Present: {', '.join(positive)}")
                    if uncertain:
                        st.markdown(f"- ⚠ Uncertain: {', '.join(uncertain)}")
                    if negative:
                        st.markdown(f"- ✗ Explicitly absent: {', '.join(negative)}")
                    
                    st.markdown("")
        else:
            st.info("No specific findings extracted.")
    
    with tab3:
        st.subheader("Export Options")
        
        # Export as CSV
        csv = df.to_csv()
        st.download_button(
            "📄 Download Matrix as CSV",
            data=csv,
            file_name="scene_graph_matrix.csv",
            mime="text/csv"
        )
        
        # Export as NumPy
        import io
        buffer = io.BytesIO()
        np.save(buffer, matrix)
        buffer.seek(0)
        st.download_button(
            "🔢 Download Matrix as NumPy (.npy)",
            data=buffer,
            file_name="scene_graph_matrix.npy",
            mime="application/octet-stream"
        )
        
        # Export metadata as JSON
        st.download_button(
            "📋 Download Metadata as JSON",
            data=json.dumps(metadata, indent=2, default=str),
            file_name="scene_graph_metadata.json",
            mime="application/json"
        )
        
        # Export findings as JSON
        findings_json = json.dumps(findings, indent=2)
        st.download_button(
            "📝 Download Findings as JSON",
            data=findings_json,
            file_name="scene_graph_findings.json",
            mime="application/json"
        )
    
    with tab4:
        st.subheader("Matrix Information")
        
        st.markdown(f"""
        **Matrix Dimensions:** {matrix.shape[0]} objects × {matrix.shape[1]} attributes
        
        **Value Legend:**
        - **+1**: Attribute is present
        - **0**: Attribute is absent or not mentioned
        - **-1**: Attribute presence is uncertain
        
        **Statistics:**
        - Total cells: {matrix.size}
        - Positive findings (+1): {np.sum(matrix == 1)}
        - Uncertain findings (-1): {np.sum(matrix == -1)}
        - Absent/Not mentioned (0): {np.sum(matrix == 0)}
        - Coverage: {(np.sum(matrix != 0) / matrix.size * 100):.2f}% of cells have findings
        """)
        
        st.markdown("---")
        st.markdown("**Anatomical Objects (29):**")
        st.write(metadata['objects'])
        
        st.markdown("---")
        st.markdown(f"**Attributes ({len(metadata['attributes'])}):**")
        st.write(metadata['attributes'])

else:
    st.info("👆 Click 'Generate Scene-Graph Matrix' to process a report")