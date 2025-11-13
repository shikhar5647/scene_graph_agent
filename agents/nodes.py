# agents/nodes.py (FIXED VERSION WITH DEBUGGING)
import re
import os
import json
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

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
client = genai.Client(api_key=api_key)


def _call_llm_safe(prompt: str) -> str:
    """Call the configured genai client."""
    try:
        if hasattr(client, "generate_content"):
            resp = client.generate_content(prompt)
            return getattr(resp, "text", str(resp))
    except Exception:
        pass

    try:
        models = getattr(client, "models", None)
        if models and hasattr(models, "generate_content"):
            resp = models.generate_content(model="gemini-2.0-flash-exp", contents=prompt)
            return getattr(resp, "text", str(resp))
    except Exception:
        pass

    try:
        if hasattr(client, "generate"):
            resp = client.generate(prompt)
            return getattr(resp, "text", str(resp))
    except Exception as e:
        raise RuntimeError("No compatible genai client method found: " + str(e))


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
    
    print(f"[DEBUG] Split into {len(sentences)} sentences")
    state["sentences"] = sentences
    return {"sentences": sentences}


def candidate_extractor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract candidate object mentions - ENHANCED VERSION."""
    sentences: List[str] = state.get("sentences", [])
    candidate_map = {o: [] for o in SG_OBJECTS}
    
    # Improved matching with keyword expansion
    object_keywords = {
        "left lung": ["left lung", "left side", "left hemithorax"],
        "right lung": ["right lung", "right side", "right hemithorax"],
        "cardiac silhouette": ["cardiac silhouette", "heart", "cardiac"],
        "left lower lung zone": ["left lower lung zone", "left lower lobe", "left base"],
        "right lower lung zone": ["right lower lung zone", "right lower lobe", "right base"],
        "left mid lung zone": ["left mid lung zone", "left middle"],
        "right mid lung zone": ["right mid lung zone", "right middle", "right mid"],
        "left upper lung zone": ["left upper lung zone", "left upper lobe", "left apex"],
        "right upper lung zone": ["right upper lung zone", "right upper lobe", "right apex"],
        "left costophrenic angle": ["left costophrenic", "left cp angle"],
        "right costophrenic angle": ["right costophrenic", "right cp angle"],
        "mediastinum": ["mediastinum", "mediastinal"],
        "spine": ["spine", "vertebra", "osseous"],
        "left hemidiaphragm": ["left hemidiaphragm", "left diaphragm"],
        "right hemidiaphragm": ["right hemidiaphragm", "right diaphragm"],
    }
    
    for s in sentences:
        s_low = s.lower()
        matched_objects = set()
        
        # Try keyword matching
        for obj, keywords in object_keywords.items():
            for keyword in keywords:
                if keyword in s_low:
                    candidate_map[obj].append(s)
                    matched_objects.add(obj)
                    break
        
        # Fallback: direct substring match for all objects
        for obj in SG_OBJECTS:
            if obj not in matched_objects and obj.lower() in s_low:
                candidate_map[obj].append(s)
                matched_objects.add(obj)
        
        # Special handling for "pleural effusion" - affects costophrenic angles
        if "pleural effusion" in s_low or "effusion" in s_low:
            if "left costophrenic angle" not in matched_objects:
                candidate_map["left costophrenic angle"].append(s)
            if "right costophrenic angle" not in matched_objects:
                candidate_map["right costophrenic angle"].append(s)
    
    # Keep only objects with sentences
    candidate_map = {k: list(set(v)) for k, v in candidate_map.items() if v}
    
    print(f"[DEBUG] Found candidates for {len(candidate_map)} objects:")
    for obj, phrases in candidate_map.items():
        print(f"  - {obj}: {len(phrases)} phrases")
    
    state["candidates"] = candidate_map
    return {"candidates": candidate_map}


def llm_enricher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract attributes using LLM - SIMPLIFIED & ROBUST."""
    candidates: Dict[str, List[str]] = state.get("candidates", {})
    if not candidates:
        print("[DEBUG] No candidates to enrich")
        return {"findings_dict": {}}
    
    findings_dict = {}
    
    for bbox_name, phrases in candidates.items():
        # Create focused prompt
        prompt = f"""Extract radiology findings for: {bbox_name}

Report sentences:
{chr(10).join(f"- {p}" for p in phrases)}

For this anatomical region, identify which attributes apply:
- Use ONLY these attributes: {', '.join(SG_ATTRIBUTES[:30])}  (showing first 30)
- Return format: {{"attribute_name": value}}
- Values: +1 (present), 0 (explicitly absent), -1 (uncertain/suspicious)

Examples:
- "normal in size" → {{"normal": 1, "enlarged": 0}}
- "no pleural effusion" → {{"pleural effusion": 0}}
- "suspicious for consolidation" → {{"consolidation": -1}}
- "patchy opacities" → {{"opacity": 1, "patchy": 1}}

Return ONLY a JSON object with attributes and values. No explanation.
"""
        
        try:
            text = _call_llm_safe(prompt)
            print(f"[DEBUG] LLM response for {bbox_name}:")
            print(f"  {text[:200]}...")
            
            # Extract JSON more robustly
            text = text.strip()
            # Remove markdown code blocks
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            
            # Find JSON object
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
            if match:
                json_text = match.group(0)
                attr_dict = json.loads(json_text)
                
                # Normalize keys to lowercase
                attr_dict = {k.lower().strip(): v for k, v in attr_dict.items()}
                
                findings_dict[bbox_name] = attr_dict
                print(f"  ✓ Extracted {len(attr_dict)} attributes")
            else:
                print(f"  ✗ No JSON found")
                findings_dict[bbox_name] = {}
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            findings_dict[bbox_name] = {}
    
    print(f"[DEBUG] Enrichment complete: {len(findings_dict)} objects with findings")
    state["findings_dict"] = findings_dict
    return {"findings_dict": findings_dict}


def llm_verifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Verify findings - LIGHTWEIGHT VERSION."""
    findings = state.get("findings_dict", {})
    if not findings:
        return {"verified_findings": {}}
    
    verified = {}
    
    # Simple verification: ensure values are -1, 0, or 1
    for obj_name, attr_dict in findings.items():
        verified_attrs = {}
        for attr, value in attr_dict.items():
            try:
                val = int(value)
                if val in [-1, 0, 1]:
                    verified_attrs[attr] = val
                elif val > 0:
                    verified_attrs[attr] = 1
                elif val < 0:
                    verified_attrs[attr] = -1
            except:
                pass
        
        if verified_attrs:
            verified[obj_name] = verified_attrs
    
    print(f"[DEBUG] Verified {len(verified)} objects")
    state["verified_findings"] = verified
    return {"verified_findings": verified}


def matrix_builder_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build matrix with FLEXIBLE attribute matching."""
    findings: Dict[str, Dict[str, int]] = state.get("verified_findings", {})
    
    matrix = np.zeros((NUM_OBJECTS, NUM_ATTRIBUTES), dtype=np.int8)
    
    # Track what we matched
    matched_count = 0
    unmatched_attrs = []
    
    for obj_name, attr_dict in findings.items():
        if obj_name not in OBJECT_TO_IDX:
            print(f"[DEBUG] Unknown object: {obj_name}")
            continue
        
        obj_idx = OBJECT_TO_IDX[obj_name]
        
        for attr_name, value in attr_dict.items():
            attr_name_clean = attr_name.lower().strip()
            
            # Direct match
            if attr_name_clean in ATTRIBUTE_TO_IDX:
                attr_idx = ATTRIBUTE_TO_IDX[attr_name_clean]
                matrix[obj_idx, attr_idx] = value
                matched_count += 1
                continue
            
            # Fuzzy match: check if attr_name contains any known attribute
            matched = False
            for known_attr in SG_ATTRIBUTES:
                if known_attr in attr_name_clean or attr_name_clean in known_attr:
                    attr_idx = ATTRIBUTE_TO_IDX[known_attr]
                    matrix[obj_idx, attr_idx] = value
                    matched_count += 1
                    matched = True
                    break
            
            if not matched:
                unmatched_attrs.append(attr_name_clean)
    
    print(f"[DEBUG] Matrix populated: {matched_count} cells")
    print(f"[DEBUG] Non-zero entries: {np.count_nonzero(matrix)}")
    if unmatched_attrs:
        print(f"[DEBUG] Unmatched attributes: {set(unmatched_attrs)}")
    
    state["scene_graph_matrix"] = matrix
    return {"scene_graph_matrix": matrix}


def aggregator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Final aggregation."""
    matrix = state.get("scene_graph_matrix")
    findings = state.get("verified_findings", {})
    
    if matrix is None:
        matrix = np.zeros((NUM_OBJECTS, NUM_ATTRIBUTES), dtype=np.int8)
    
    metadata = {
        "objects": SG_OBJECTS,
        "attributes": SG_ATTRIBUTES,
        "matrix_shape": matrix.shape,
        "value_legend": {
            "+1": "present",
            "0": "absent/not mentioned",
            "-1": "uncertain"
        },
        "findings_summary": findings,
        "statistics": {
            "total_cells": int(matrix.size),
            "positive": int(np.sum(matrix == 1)),
            "negative": int(np.sum(matrix == 0)),
            "uncertain": int(np.sum(matrix == -1)),
            "coverage": float(np.sum(matrix != 0) / matrix.size * 100)
        }
    }
    
    print(f"[DEBUG] Final matrix shape: {matrix.shape}")
    print(f"[DEBUG] Statistics: {metadata['statistics']}")
    
    state["final_matrix"] = matrix
    state["metadata"] = metadata
    
    return {
        "final_matrix": matrix,
        "metadata": metadata
    }