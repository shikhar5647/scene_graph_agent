# app/streamlit_app.py - SGRRG Matrix Interface
import streamlit as st
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from agents.graph import run_graph

st.set_page_config(page_title="SGRRG Scene Graph Matrix", layout="wide", initial_sidebar_state="expanded")

st.title("🏥 Radiology Report → SGRRG Scene Graph Matrix")

st.markdown("""
Generate a **29 objects × 180 attributes** scene graph matrix from chest X-ray reports following the 
[SGRRG paper](https://arxiv.org/pdf/2108.00316) schema.

**Matrix Values:**
- **+1** = Attribute present
- **0** = Attribute explicitly absent  
- **-1** = Attribute uncertain (suspicious/possible)
""")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **SGRRG Schema:**
    - 29 anatomical objects
    - ~180 attributes in 9 categories:
        - Anatomical findings
        - Diseases
        - Tubes & lines
        - Devices
        - Technical assessment
        - NLP descriptors
        - Severity
        - Temporal
        - Texture
    
    **Citation:**
    Scene Graph Aided Radiology Report Generation (SGRRG)
    """)

# Load sample
sample_path = Path(__file__).parent.parent / "examples" / "sample_report.txt"
default_text = sample_path.read_text() if sample_path.exists() else ""

report_text = st.text_area("📝 Radiology Report", value=default_text, height=300)

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if st.button("🚀 Generate Scene Graph Matrix", type="primary"):
        if not report_text.strip():
            st.warning("Please enter a radiology report.")
        else:
            with st.spinner("Extracting findings and building matrix..."):
                try:
                    result = run_graph(report_text)
                    st.session_state['result'] = result
                    st.success("✅ Matrix generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

with col2:
    if st.button("🗑️ Clear Results"):
        if 'result' in st.session_state:
            del st.session_state['result']
        st.rerun()

# Display results
if 'result' in st.session_state:
    result = st.session_state['result']
    matrix = result["matrix"]
    metadata = result["metadata"]
    
    st.divider()
    
    # Statistics
    stats = metadata['statistics']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Matrix Size", f"{matrix.shape[0]}×{matrix.shape[1]}")
    with col2:
        st.metric("Present (+1)", stats['positive'])
    with col3:
        st.metric("Absent (0)", stats['negative'])
    with col4:
        st.metric("Uncertain (-1)", stats['uncertain'])
    with col5:
        st.metric("Coverage", f"{stats['coverage']:.1f}%")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Matrix View", 
        "🔍 Findings by Object", 
        "📋 Findings by Category",
        "💾 Export",
        "📖 Schema Info"
    ])
    
    with tab1:
        st.subheader("Scene Graph Matrix")
        
        objects = metadata['objects']
        attributes = metadata['attributes']
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_nonzero_only = st.checkbox("Show only objects with findings", value=True)
        with col2:
            filter_category = st.selectbox("Filter by attribute category", 
                                          ["All"] + list(set(metadata['attribute_categories'].values())))
        with col3:
            value_filter = st.selectbox("Filter by value", ["All", "Present (+1)", "Absent (0)", "Uncertain (-1)"])
        
        # Build filtered dataframe
        df = pd.DataFrame(matrix, index=objects, columns=attributes)
        
        # Filter attributes by category
        if filter_category != "All":
            attr_cats = metadata['attribute_categories']
            cols_to_keep = [a for a in attributes if attr_cats.get(a) == filter_category]
            df = df[cols_to_keep]
        
        # Filter rows
        if show_nonzero_only:
            mask = (df != 0).any(axis=1)
            df = df[mask]
        
        # Filter by value
        if value_filter != "All":
            val_map = {"Present (+1)": 1, "Absent (0)": 0, "Uncertain (-1)": -1}
            target_val = val_map[value_filter]
            cols_with_val = (df == target_val).any(axis=0)
            df = df.loc[:, cols_with_val]
        
        # Show only columns with non-zero values
        df = df.loc[:, (df != 0).any(axis=0)]
        
        # Style the dataframe
        def style_cell(val):
            if val == 1:
                return 'background-color: #d4edda; color: #155724; font-weight: bold'
            elif val == -1:
                return 'background-color: #fff3cd; color: #856404; font-weight: bold'
            elif val == 0:
                return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
            elif val == -2:
                return 'background-color: #f8f9fa; color: #6c757d'
            return ''
        
        if not df.empty:
            st.dataframe(
                df.style.applymap(style_cell),
                use_container_width=True,
                height=500
            )
            st.caption(f"Showing {df.shape[0]} objects × {df.shape[1]} attributes")
        else:
            st.info("No findings match the current filters.")
    
    with tab2:
        st.subheader("Findings Organized by Anatomical Object")
        
        findings = metadata.get('findings_summary', {})
        attr_cats = metadata['attribute_categories']
        
        if findings:
            # Show each object with findings
            for obj_name in sorted(findings.keys()):
                attrs = findings[obj_name]
                if not attrs:
                    continue
                
                with st.expander(f"**{obj_name}** ({len(attrs)} findings)", expanded=False):
                    # Group by category
                    by_category = {}
                    for attr, val in attrs.items():
                        cat = attr_cats.get(attr, 'unknown')
                        if cat not in by_category:
                            by_category[cat] = []
                        
                        symbol = {1: "✓", 0: "✗", -1: "⚠"}.get(val, "?")
                        color = {1: "green", 0: "red", -1: "orange"}.get(val, "gray")
                        by_category[cat].append((symbol, attr, val, color))
                    
                    # Display by category
                    for cat in sorted(by_category.keys()):
                        st.markdown(f"**{cat}**")
                        for symbol, attr, val, color in by_category[cat]:
                            st.markdown(f":{color}[{symbol}] {attr} ({val:+d})")
                        st.markdown("")
        else:
            st.info("No findings extracted.")
    
    with tab3:
        st.subheader("Findings Organized by Attribute Category")
        
        # Aggregate by category
        attr_cats = metadata['attribute_categories']
        category_findings = {}
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] != 0:
                    obj = objects[i]
                    attr = attributes[j]
                    val = matrix[i, j]
                    cat = attr_cats.get(attr, 'unknown')
                    
                    if cat not in category_findings:
                        category_findings[cat] = []
                    
                    category_findings[cat].append((obj, attr, val))
        
        # Display
        for cat in sorted(category_findings.keys()):
            with st.expander(f"**{cat}** ({len(category_findings[cat])} findings)", expanded=False):
                findings_list = category_findings[cat]
                
                # Group by value
                for val in [1, -1, 0]:
                    items = [(o, a) for o, a, v in findings_list if v == val]
                    if items:
                        val_label = {1: "✓ Present", -1: "⚠ Uncertain", 0: "✗ Absent"}[val]
                        st.markdown(f"**{val_label}** ({len(items)})")
                        for obj, attr in items:
                            st.markdown(f"- {obj}: {attr}")
                        st.markdown("")
    
    with tab4:
        st.subheader("Export Options")
        
        # CSV
        csv = pd.DataFrame(matrix, index=objects, columns=attributes).to_csv()
        st.download_button(
            "📄 Download Matrix as CSV",
            data=csv,
            file_name="sgrrg_scene_graph.csv",
            mime="text/csv"
        )
        
        # NumPy
        import io
        buffer = io.BytesIO()
        np.save(buffer, matrix)
        buffer.seek(0)
        st.download_button(
            "🔢 Download Matrix as NumPy (.npy)",
            data=buffer,
            file_name="sgrrg_scene_graph.npy",
            mime="application/octet-stream"
        )
        
        # Metadata JSON
        st.download_button(
            "📋 Download Metadata (JSON)",
            data=json.dumps(metadata, indent=2, default=str),
            file_name="sgrrg_metadata.json",
            mime="application/json"
        )
        
        # Findings JSON
        findings_json = {
            "report": report_text,
            "findings": metadata.get('findings_summary', {}),
            "statistics": stats
        }
        st.download_button(
            "📝 Download Findings Summary (JSON)",
            data=json.dumps(findings_json, indent=2),
            file_name="sgrrg_findings.json",
            mime="application/json"
        )
    
    with tab5:
        st.subheader("SGRRG Schema Information")
        
        st.markdown("### 29 Anatomical Objects")
        st.write(objects)
        
        st.markdown(f"### {len(attributes)} Attributes")
        
        # Show attributes by category
        attr_cats = metadata['attribute_categories']
        cats = {}
        for attr in attributes:
            cat = attr_cats.get(attr, 'unknown')
            if cat not in cats:
                cats[cat] = []
            cats[cat].append(attr)
        
        for cat in sorted(cats.keys()):
            with st.expander(f"{cat} ({len(cats[cat])} attributes)"):
                st.write(cats[cat])

else:
    st.info("👆 Enter a radiology report and click 'Generate Scene Graph Matrix' to begin.")