# scene_graph_agent

Radiology Report → Scene-Graph (Chest ImaGenome / SGRRG style)

This repository converts plain radiology report text (e.g., chest x-ray findings + impression)
into a Chest-ImaGenome / SGRRG-style scene-graph JSON mapping 29 anatomical objects to
normalized attributes, using a small LangGraph-like pipeline and the Gemini (Google GenAI)
model for enrichment and verification.

What's included
- agents/
	- `nodes.py` : node functions that do sentence-splitting, candidate extraction, LLM enrichment, verification and aggregation
	- `graph.py` : runs the pipeline; falls back to a sequential runner if the installed `langgraph` is incompatible
- app/
	- `streamlit_app.py` : Streamlit UI to paste a report and produce/download the scene-graph JSON
- utils/
	- `config.py` : list of 29 anatomical objects and category ids
	- `prompts.py` : prompt template used to instruct the model

Quick start
1) Create a Python virtual environment and activate it (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install requirements:

```powershell
pip install -r requirements.txt
```

3) Create a `.env` file in the project root and set your Gemini API key (example):

```
GEMINI_API_KEY="your_gemini_api_key_here"
```

4) Run the Streamlit app from the project root:

```powershell
cd path\to\scene_graph_agent
streamlit run app/streamlit_app.py
```

Notes
- The code tries to initialize `genai.Client(api_key=...)` and uses a compatibility wrapper to call the LLM. If you still see method-not-found errors, ensure your installed GenAI SDK matches one of the supported call patterns (see `agents/nodes.py` for the details).
- The `agents/graph.py` file will attempt to use `langgraph`'s `StateGraph` if available; if there are API incompatibilities the code falls back to a sequential runner.

Sample test text
Use the `examples/sample_report.txt` file or paste the following into the app:

"""
Exam: Chest radiograph, AP upright.
Findings: The cardiac silhouette is normal in size. There are patchy airspace opacities in the right mid and right lower lung zones, greater on the right. No pleural effusion identified. Lungs are otherwise clear.
Impression: Patchy consolidation in the right mid and lower lung zones, suspicious for infectious process. No cardiomegaly. No pleural effusion.
"""

If you want me to test or adapt the LLM call pattern to your installed `google-genai`/`google-generativeai` package version, tell me the package name/version and I will update the call sites accordingly.