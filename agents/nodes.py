# agents/nodes.py
import re
from typing import Dict, Any, List
from google import genai
from utils.config import SG_OBJECTS
from utils.prompts import BASE_ENRICH_PROMPT

# Initialize Gemini client once (client picks up GEMINI_API_KEY from env)
client = genai.Client()

# Simple sentence splitter - keeps sentence boundaries naive but robust for typical reports
def split_report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    report: str = state.get("report_text", "").strip()
    if not report:
        return {"error": "no report_text provided"}
    # naive splitting by newline and punctuation (radiology reports often line-break per sentence)
    sentences = []
    for line in report.splitlines():
        line = line.strip()
        if not line:
            continue
        # split on sentence-ending punctuation but keep short fragments
        parts = re.split(r'(?<=[\.\?!])\s+', line)
        for p in parts:
            p = p.strip()
            if p:
                sentences.append(p)
    state["sentences"] = sentences
    return {"sentences": sentences}

# rule-based object matcher: attempt to map sentences to candidate objects using simple substring match
def candidate_extractor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    sentences: List[str] = state.get("sentences", [])
    candidate_map = {o: [] for o in SG_OBJECTS}
    lowered_objs = [(o, o.lower()) for o in SG_OBJECTS]
    for s in sentences:
        s_low = s.lower()
        matched = False
        for obj, obj_low in lowered_objs:
            # exact substring match or split-word match
            if obj_low in s_low:
                candidate_map[obj].append(s)
                matched = True
        # fallback heuristics: look for 'left', 'right' + zone words or common terms
        # (this is purposely simple — main disambiguation is done by LLM later)
    # prune empty lists
    candidate_map = {k: v for k, v in candidate_map.items() if v}
    state["candidates"] = candidate_map
    return {"candidates": candidate_map}

# LLM enrichment node: calls Gemini 2.5 Pro to convert candidate sentences -> structured attributes (Chest-ImaGenome style)
def llm_enricher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    candidates: Dict[str, List[str]] = state.get("candidates", {})
    if not candidates:
        return {"scene_graph": {}}
    # We'll craft a prompt that injects the exact sentences
    # For reliability, handle each object independently (smaller prompts)
    scene_graph = {}
    for bbox_name, phrases in candidates.items():
        prompt = BASE_ENRICH_PROMPT + "\n\n"
        prompt += f"Target object: {bbox_name}\n\n"
        prompt += "Report sentences associated:\n"
        for p in phrases:
            prompt += f"- {p}\n"
        prompt += "\nProduce a JSON object for this single bounding object with fields: bbox_name, attributes (list of lists), phrases (list). Output only the JSON for the object.\n"
        # call Gemini 2.5 Pro
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt
            )
            text = response.text if hasattr(response, "text") else response
            # The model should return JSON; attempt to parse it conservatively
            import json
            # try to extract the first JSON block
            m = re.search(r'(\{(?:.|\n)*\})', text)
            if m:
                json_text = m.group(1)
            else:
                json_text = text
            obj_dict = json.loads(json_text)
            # Basic normalization: ensure bbox_name matches
            obj_dict["bbox_name"] = bbox_name
            obj_dict.setdefault("phrases", phrases)
            scene_graph[bbox_name] = obj_dict
        except Exception as e:
            # On failure, fallback to a minimal entry
            scene_graph[bbox_name] = {
                "bbox_name": bbox_name,
                "attributes": [],
                "phrases": phrases
            }
    state["scene_graph_partial"] = scene_graph
    return {"scene_graph_partial": scene_graph}

# verifier node: ask Gemini to verify and normalize the scene graph JSON (single consolidated pass)
def llm_verifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    partial = state.get("scene_graph_partial", {})
    if not partial:
        return {"scene_graph": {}}
    import json
    prompt = (
        "You are an assistant that receives a partial Chest-ImaGenome-style scene graph (JSON). "
        "Validate the JSON, normalize attribute labels (lowercase, short phrases), and remove obviously contradictory entries. "
        "Return the final scene_graph JSON mapping bbox_name -> object-dictionary. Only output JSON.\n\n"
        "Input JSON:\n"
        + json.dumps(partial, indent=2) +
        "\n\nOutput the corrected JSON."
    )
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt
        )
        text = response.text if hasattr(response, "text") else response
        import re, json
        m = re.search(r'(\{(?:.|\n)*\})', text)
        if m:
            json_text = m.group(1)
        else:
            json_text = text
        final = json.loads(json_text)
    except Exception as e:
        # fallback to original partial
        final = partial
    state["scene_graph"] = final
    return {"scene_graph": final}

# aggregator node: final formatting (if needed)
def aggregator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    sg = state.get("scene_graph", {})
    # ensure top-level mapping format similar to Chest ImaGenome: each value contains attributes, phrases, etc.
    # (we assume verifier returned that)
    return {"scene_graph": sg}
