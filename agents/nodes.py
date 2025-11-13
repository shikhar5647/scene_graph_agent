# agents/nodes.py
import re
import os
import numpy as np
from typing import Dict, Any, List
from dotenv import load_dotenv
from google import genai
from utils.config import (
    SG_OBJECTS, SG_ATTRIBUTES, 
    OBJECT_TO_IDX, ATTRIBUTE_TO_IDX,
    NUM_OBJECTS, NUM_ATTRIBUTES
)
from utils.prompts import BASE_ENRICH_PROMPT, VERIFICATION_PROMPT

# Load environment variables from .env file
load_dotenv()

# Initialize Gemini client with API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
client = genai.Client(api_key=api_key)


def _call_llm_safe(prompt: str) -> str:
    """Call the configured genai client with a variety of possible method names."""
    # 1) direct generate_content on client
    try:
        if hasattr(client, "generate_content"):
            resp = client.generate_content(prompt)
            return getattr(resp, "text", str(resp))
    except Exception:
        pass

    # 2) client.models.generate_content(model=..., contents=...)
    try:
        models = getattr(client, "models", None)
        if models and hasattr(models, "generate_content"):
            resp = models.generate_content(model="gemini-2.0-flash-exp", contents=prompt)
            return getattr(resp, "text", str(resp))
    except Exception:
        pass

    # 3) older client.generate or client.generate_text
    try:
        if hasattr(client, "generate"):
            resp = client.generate(prompt)
            return getattr(resp, "text", str(resp))
        if hasattr(client, "generate_text"):
            resp = client.generate_text(prompt)
            return getattr(resp, "text", str(resp))
    except Exception:
        pass

    # 4) last-resort: try to call __call__
    try:
        resp = client(prompt)
        return getattr(resp, "text", str(resp))
    except Exception as e:
        raise RuntimeError("No compatible genai client method found or call failed: " + str(e))


def split_report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Split report into sentences."""
    report: str = state.get("report_text", "").strip()
    if not report:
        return {"error": "no report_text provided"}
    
    sentences = []
    for line in report.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r'(?<=[\.\?!])\s+', line)
        for p in parts:
            p = p.strip()
            if p:
                sentences.append(p)
    
    state["sentences"] = sentences
    return {"sentences": sentences}


def candidate_extractor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract candidate object mentions from sentences."""
    sentences: List[str] = state.get("sentences", [])
    candidate_map = {o: [] for o in SG_OBJECTS}
    lowered_objs = [(o, o.lower()) for o in SG_OBJECTS]
    
    for s in sentences:
        s_low = s.lower()
        for obj, obj_low in lowered_objs:
            if obj_low in s_low:
                candidate_map[obj].append(s)
    
    # Keep only objects with associated sentences
    candidate_map = {k: v for k, v in candidate_map.items() if v}
    state["candidates"] = candidate_map
    return {"candidates": candidate_map}


def llm_enricher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Use LLM to extract attributes for each object mention."""
    candidates: Dict[str, List[str]] = state.get("candidates", {})
    if not candidates:
        return {"findings_dict": {}}
    
    findings_dict = {}
    
    for bbox_name, phrases in candidates.items():
        prompt = BASE_ENRICH_PROMPT + "\n\n"
        prompt += f"Target anatomical object: {bbox_name}\n\n"
        prompt += "Report sentences mentioning this object:\n"
        for p in phrases:
            prompt += f"- {p}\n"
        prompt += "\nExtract attributes and their status (+1, 0, or -1) for this object. Output only JSON.\n"
        
        try:
            text = _call_llm_safe(prompt)
            import json
            # Extract JSON from response
            m = re.search(r'(\{(?:.|\n)*\})', text)
            if m:
                json_text = m.group(1)
            else:
                json_text = text
            
            obj_findings = json.loads(json_text)
            
            # Ensure it's in the right format: {bbox_name: {attr: value}}
            if bbox_name in obj_findings:
                findings_dict[bbox_name] = obj_findings[bbox_name]
            else:
                findings_dict[bbox_name] = obj_findings
                
        except Exception as e:
            # On failure, create empty entry
            findings_dict[bbox_name] = {}
    
    state["findings_dict"] = findings_dict
    return {"findings_dict": findings_dict}


def llm_verifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Verify and normalize the extracted findings."""
    findings = state.get("findings_dict", {})
    if not findings:
        return {"verified_findings": {}}
    
    import json
    prompt = VERIFICATION_PROMPT + "\n\n"
    prompt += "Input findings JSON:\n"
    prompt += json.dumps(findings, indent=2)
    prompt += "\n\nOutput the verified and corrected JSON."
    
    try:
        text = _call_llm_safe(prompt)
        m = re.search(r'(\{(?:.|\n)*\})', text)
        if m:
            json_text = m.group(1)
        else:
            json_text = text
        verified = json.loads(json_text)
    except Exception:
        # Fallback to original findings
        verified = findings
    
    state["verified_findings"] = verified
    return {"verified_findings": verified}


def matrix_builder_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build the objects × attributes matrix from verified findings."""
    findings: Dict[str, Dict[str, int]] = state.get("verified_findings", {})
    
    # Initialize matrix with zeros (absent/not mentioned)
    matrix = np.zeros((NUM_OBJECTS, NUM_ATTRIBUTES), dtype=np.int8)
    
    # Populate matrix based on findings
    for obj_name, attr_dict in findings.items():
        if obj_name not in OBJECT_TO_IDX:
            continue
        
        obj_idx = OBJECT_TO_IDX[obj_name]
        
        for attr_name, value in attr_dict.items():
            # Normalize attribute name (lowercase, strip)
            attr_name_normalized = attr_name.lower().strip()
            
            if attr_name_normalized not in ATTRIBUTE_TO_IDX:
                # Try to find close match
                for known_attr in SG_ATTRIBUTES:
                    if known_attr in attr_name_normalized or attr_name_normalized in known_attr:
                        attr_name_normalized = known_attr
                        break
            
            if attr_name_normalized in ATTRIBUTE_TO_IDX:
                attr_idx = ATTRIBUTE_TO_IDX[attr_name_normalized]
                # Ensure value is -1, 0, or +1
                if value in [-1, 0, 1]:
                    matrix[obj_idx, attr_idx] = value
    
    state["scene_graph_matrix"] = matrix
    return {"scene_graph_matrix": matrix}


def aggregator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Final aggregation - prepare matrix and metadata for output."""
    matrix = state.get("scene_graph_matrix")
    findings = state.get("verified_findings", {})
    
    if matrix is None:
        matrix = np.zeros((NUM_OBJECTS, NUM_ATTRIBUTES), dtype=np.int8)
    
    # Create metadata for easy interpretation
    metadata = {
        "objects": SG_OBJECTS,
        "attributes": SG_ATTRIBUTES,
        "matrix_shape": matrix.shape,
        "value_legend": {
            "+1": "present",
            "0": "absent/not mentioned",
            "-1": "uncertain"
        },
        "findings_summary": findings
    }
    
    state["final_matrix"] = matrix
    state["metadata"] = metadata
    
    return {
        "final_matrix": matrix,
        "metadata": metadata
    }