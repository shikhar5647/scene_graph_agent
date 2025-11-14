# agents/nodes.py - FIXED: Rule-based + LLM Hybrid Approach
import re
import os
import json
import numpy as np
from typing import Dict, Any, List
from dotenv import load_dotenv
from utils.config import (
    SG_OBJECTS, SG_ATTRIBUTES, ATTRIBUTE_CATEGORIES,
    OBJECT_TO_IDX, ATTRIBUTE_TO_IDX,
    NUM_OBJECTS, NUM_ATTRIBUTES
)

load_dotenv()

# Initialize Gemini
try:
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        client = genai.Client(api_key=api_key)
        LLM_AVAILABLE = True
        print("[INFO] ✓ Gemini client initialized")
    else:
        LLM_AVAILABLE = False
        print("[WARNING] No API key - using rule-based extraction only")
except Exception as e:
    LLM_AVAILABLE = False
    print(f"[WARNING] Gemini initialization failed: {e}")


def _call_llm_safe(prompt: str) -> str:
    """Call LLM with proper error handling."""
    if not LLM_AVAILABLE:
        return ""
    
    try:
        # Try method 1: direct generate_content
        if hasattr(client, "generate_content"):
            resp = client.generate_content(prompt)
            text = getattr(resp, "text", str(resp))
            if text:
                return text
    except Exception as e:
        print(f"[LLM] Method 1 failed: {e}")
    
    try:
        # Try method 2: models.generate_content
        models = getattr(client, "models", None)
        if models and hasattr(models, "generate_content"):
            resp = models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt
            )
            text = getattr(resp, "text", str(resp))
            if text:
                return text
    except Exception as e:
        print(f"[LLM] Method 2 failed: {e}")
    
    return ""


# ============================================================================
# RULE-BASED EXTRACTION (Baseline)
# ============================================================================

def extract_findings_rule_based(text: str, object_name: str) -> Dict[str, int]:
    """
    Rule-based extraction returning {attribute: value}.
    Values: 1 (present), 0 (explicitly absent), -1 (uncertain), -2 (not mentioned)
    """
    text_lower = text.lower()
    findings = {}
    
    # POSITIVE patterns
    positive_patterns = {
        'normal': [r'\bnormal\b', r'unremarkable', r'within normal limits'],
        'clear': [r'\bclear\b', r'well aerated'],
        'hyperinflated': [r'hyperinflat'],
        'elevated': [r'elevated', r'elevation'],
        'flattened': [r'flattened'],
        'displaced': [r'displaced'],
        'enlarged': [r'enlarged', r'enlargement'],
        'tortuous': [r'tortuous'],
        'opacity': [r'opacit(?:y|ies)', r'densit(?:y|ies)'],
        'consolidation': [r'consolidation', r'consolidate'],
        'infiltrate': [r'infiltrat'],
        'atelectasis': [r'atelectasis'],
        'collapse': [r'\bcollapse\b', r'collapsed'],
        'pleural effusion': [r'pleural\s+effusion', r'\beffusion\b'],
        'pneumothorax': [r'pneumothorax'],
        'pulmonary edema': [r'pulmonary\s+edema', r'\bedema\b'],
        'cardiomegaly': [r'cardiomegaly', r'cardiac\s+enlargement'],
        'mass': [r'\bmass\b'],
        'nodule': [r'nodule'],
        'calcification': [r'calcification', r'calcified'],
        'fibrosis': [r'fibrosis'],
        'scarring': [r'scar(?:ring)?'],
        'thickening': [r'thickening'],
        'pneumonia': [r'pneumonia'],
        'infection': [r'infection', r'infectious'],
        'fracture': [r'fracture'],
        'focal': [r'\bfocal\b'],
        'diffuse': [r'diffuse'],
        'patchy': [r'patchy'],
        'bilateral': [r'bilateral', r'\bboth\b'],
        'mild': [r'\bmild\b'],
        'moderate': [r'moderate'],
        'severe': [r'severe'],
        'acute': [r'\bacute\b'],
        'chronic': [r'chronic'],
    }
    
    # NEGATIVE patterns (explicitly absent)
    negative_patterns = {
        'opacity': [r'no\s+opacit', r'clear\s+of\s+opacit'],
        'consolidation': [r'no\s+consolidation', r'clear\s+of\s+consolidation'],
        'pleural effusion': [r'no\s+(?:pleural\s+)?effusion'],
        'pneumothorax': [r'no\s+pneumothorax'],
        'cardiomegaly': [r'no\s+cardiomegaly', r'normal\s+cardiac\s+size'],
        'enlarged': [r'not\s+enlarged', r'normal\s+(?:in\s+)?size'],
        'atelectasis': [r'no\s+atelectasis'],
        'pulmonary edema': [r'no\s+(?:pulmonary\s+)?edema'],
        'acute': [r'no\s+acute'],
        'fracture': [r'no\s+fracture'],
    }
    
    # UNCERTAIN patterns
    uncertain_patterns = {
        'consolidation': [r'suspicious\s+for.*consolidation', r'possible.*consolidation'],
        'opacity': [r'suspicious\s+for.*opacit', r'possible.*opacit'],
        'pneumonia': [r'suspicious\s+for.*pneumonia', r'concerning\s+for.*pneumonia'],
        'infection': [r'suspicious\s+for.*infection', r'concerning\s+for.*infectious'],
        'mass': [r'suspicious.*mass', r'possible\s+mass'],
    }
    
    # Extract uncertain first (highest priority)
    for attr, patterns in uncertain_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                findings[attr] = -1
                break
    
    # Extract negative
    for attr, patterns in negative_patterns.items():
        if attr not in findings:
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    findings[attr] = 0
                    break
    
    # Extract positive
    for attr, patterns in positive_patterns.items():
        if attr not in findings:
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    findings[attr] = 1
                    break
    
    # Object-specific logic
    obj_lower = object_name.lower()
    if 'cardiac' in obj_lower:
        if findings.get('normal') == 1:
            findings['enlarged'] = 0
            findings['cardiomegaly'] = 0
    
    if 'lung' in obj_lower:
        if findings.get('clear') == 1:
            findings.setdefault('consolidation', 0)
            findings.setdefault('opacity', 0)
    
    return findings


# ============================================================================
# LLM EXTRACTION (Enhanced)
# ============================================================================

def extract_findings_llm(text: str, object_name: str, rule_findings: Dict[str, int]) -> Dict[str, int]:
    """
    Use LLM to extract findings, using rule-based results as context.
    """
    if not LLM_AVAILABLE:
        return {}
    
    # Create attribute list focused on report
    relevant_attrs = [a for a in SG_ATTRIBUTES[:50]]  # Use top 50 most common
    attr_list = ", ".join(relevant_attrs)
    
    prompt = f"""You are a medical NLP expert analyzing chest X-ray reports.

TASK: Extract structured findings for this anatomical region: "{object_name}"

REPORT TEXT:
{text}

RULE-BASED FINDINGS (for reference):
{json.dumps(rule_findings, indent=2)}

INSTRUCTIONS:
1. Analyze the report text for findings related to {object_name}
2. Return ONLY attributes that are EXPLICITLY mentioned or clearly implied
3. Use these values:
   - 1 = Definitely present (e.g., "opacity", "cardiomegaly")
   - 0 = Explicitly absent (e.g., "no effusion", "no consolidation")
   - -1 = Uncertain (e.g., "suspicious for", "possible", "may represent")
   - DO NOT include -2 (that's for attributes not mentioned at all)

4. Common attributes to consider: {attr_list}

OUTPUT FORMAT (JSON only, no explanation):
{{
  "opacity": 1,
  "patchy": 1,
  "consolidation": -1,
  "pleural effusion": 0
}}

Return ONLY the JSON object:"""
    
    try:
        response = _call_llm_safe(prompt)
        if not response:
            return {}
        
        print(f"[LLM] Response for {object_name}: {response[:150]}...")
        
        # Clean response
        response = re.sub(r'```(?:json)?', '', response)
        response = response.strip()
        
        # Extract JSON
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\})*[^{}]*\}', response)
        if match:
            json_str = match.group(0)
            llm_findings = json.loads(json_str)
            
            # Normalize
            normalized = {}
            for key, val in llm_findings.items():
                key_clean = key.lower().strip()
                if isinstance(val, (int, float)) and val in [-1, 0, 1]:
                    normalized[key_clean] = int(val)
            
            print(f"[LLM] Extracted {len(normalized)} findings")
            return normalized
        else:
            print(f"[LLM] No JSON found in response")
            return {}
            
    except Exception as e:
        print(f"[LLM] Extraction failed: {e}")
        return {}


# ============================================================================
# PIPELINE NODES
# ============================================================================

def split_report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Split report into sentences."""
    report = state.get("report_text", "").strip()
    if not report:
        return {"error": "no report_text provided"}
    
    sentences = []
    for line in report.splitlines():
        line = line.strip()
        if not line or line.startswith('Exam:'):
            continue
        parts = re.split(r'(?<!\d)\.(?!\d)\s+', line)
        for p in parts:
            p = p.strip()
            if p and len(p) > 5:
                sentences.append(p)
    
    print(f"\n[SPLIT] {len(sentences)} sentences")
    state["sentences"] = sentences
    return {"sentences": sentences}


def candidate_extractor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract object mentions."""
    sentences = state.get("sentences", [])
    candidate_map = {}
    
    object_patterns = {
        "cardiac silhouette": [r'cardiac(?:\s+silhouette)?', r'\bheart\b', r'cardiomegaly'],
        "left lung": [r'left\s+lung', r'left\s+hemithorax', r'on\s+the\s+left(?!\s+(?:lower|mid|upper))'],
        "right lung": [r'right\s+lung', r'right\s+hemithorax', r'on\s+the\s+right(?!\s+(?:lower|mid|upper))'],
        "left lower lung zone": [r'left\s+lower\s+(?:lung\s+)?(?:zone|lobe)', r'left\s+base'],
        "right lower lung zone": [r'right\s+lower\s+(?:lung\s+)?(?:zone|lobe)', r'right\s+base'],
        "left mid lung zone": [r'left\s+mid(?:dle)?\s+(?:lung\s+)?zone'],
        "right mid lung zone": [r'right\s+mid(?:dle)?\s+(?:lung\s+)?zone'],
        "left costophrenic angle": [r'left\s+costophrenic'],
        "right costophrenic angle": [r'right\s+costophrenic'],
        "spine": [r'spine', r'vertebra', r'osseous'],
    }
    
    for s in sentences:
        s_lower = s.lower()
        for obj_name, patterns in object_patterns.items():
            for pattern in patterns:
                if re.search(pattern, s_lower):
                    if obj_name not in candidate_map:
                        candidate_map[obj_name] = []
                    if s not in candidate_map[obj_name]:
                        candidate_map[obj_name].append(s)
                    break
        
        # Special cases
        if re.search(r'pleural\s+effusion|effusion', s_lower):
            for obj in ["left costophrenic angle", "right costophrenic angle"]:
                if obj not in candidate_map:
                    candidate_map[obj] = []
                if s not in candidate_map[obj]:
                    candidate_map[obj].append(s)
        
        if re.search(r'\blungs\b', s_lower) and not re.search(r'left|right', s_lower):
            for obj in ["left lung", "right lung"]:
                if obj not in candidate_map:
                    candidate_map[obj] = []
                if s not in candidate_map[obj]:
                    candidate_map[obj].append(s)
    
    print(f"[EXTRACT] {len(candidate_map)} objects identified")
    state["candidates"] = candidate_map
    return {"candidates": candidate_map}


def llm_enricher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: Actually call LLM and merge with rule-based findings.
    """
    candidates = state.get("candidates", {})
    if not candidates:
        return {"findings_dict": {}}
    
    findings_dict = {}
    
    for obj_name, phrases in candidates.items():
        combined_text = " ".join(phrases)
        
        print(f"\n[PROCESS] {obj_name}")
        print(f"  Text: {combined_text[:100]}...")
        
        # 1. Rule-based extraction (baseline)
        rule_findings = extract_findings_rule_based(combined_text, obj_name)
        print(f"  Rule-based: {len(rule_findings)} findings")
        
        # 2. LLM extraction (enhancement)
        llm_findings = extract_findings_llm(combined_text, obj_name, rule_findings)
        print(f"  LLM: {len(llm_findings)} findings")
        
        # 3. Merge: LLM takes precedence, then rules
        merged = rule_findings.copy()
        for attr, val in llm_findings.items():
            # LLM overrides rule-based
            merged[attr] = val
        
        findings_dict[obj_name] = merged
        print(f"  Final: {len(merged)} findings")
    
    state["findings_dict"] = findings_dict
    return {"findings_dict": findings_dict}


def llm_verifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: Actually verify findings using LLM.
    """
    findings = state.get("findings_dict", {})
    if not findings or not LLM_AVAILABLE:
        # Just validate values
        verified = {}
        for obj, attrs in findings.items():
            verified[obj] = {k: v for k, v in attrs.items() if v in [-1, 0, 1]}
        state["verified_findings"] = verified
        return {"verified_findings": verified}
    
    print(f"\n[VERIFY] Validating {len(findings)} objects...")
    
    # Create verification prompt
    prompt = f"""You are a medical expert validating extracted radiology findings.

EXTRACTED FINDINGS:
{json.dumps(findings, indent=2)}

VALIDATION RULES:
1. Check logical consistency (e.g., can't have "normal" AND "cardiomegaly" both as 1)
2. Verify negations are correct (0 = explicitly absent)
3. Check uncertainty markers (-1 = suspicious/possible)
4. Remove contradictions
5. Values must be: 1 (present), 0 (absent), -1 (uncertain)

Return the corrected findings in the SAME JSON format. Output ONLY JSON:"""
    
    try:
        response = _call_llm_safe(prompt)
        if response:
            response = re.sub(r'```(?:json)?', '', response).strip()
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\})*[^{}]*\}', response, re.DOTALL)
            if match:
                raw_verified = json.loads(match.group(0))
                
                # --- START FIX: Normalize keys from verifier LLM ---
                verified = {}
                for obj_name, attrs in raw_verified.items():
                    clean_obj = obj_name.lower().strip()
                    if clean_obj not in verified:
                        verified[clean_obj] = {}
                    for attr_name, val in attrs.items():
                        clean_attr = attr_name.lower().strip()
                        if val in [-1, 0, 1]:
                            verified[clean_obj][clean_attr] = int(val)
                print(f"[VERIFY] ✓ Validated and normalized findings")
                state["verified_findings"] = verified
                return {"verified_findings": verified}
    except Exception as e:
        print(f"[VERIFY] Failed: {e}")
    
    # Fallback: basic validation
    verified = {}
    for obj, attrs in findings.items():
        verified[obj] = {k: v for k, v in attrs.items() if v in [-1, 0, 1]}
    
    state["verified_findings"] = verified
    return {"verified_findings": verified}


def matrix_builder_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build matrix with 4 values: -2 (unknown), -1 (uncertain), 0 (absent), 1 (present).
    """
    findings = state.get("verified_findings", {})
    
    # Initialize with -2 (unknown/not mentioned)
    matrix = np.full((NUM_OBJECTS, NUM_ATTRIBUTES), -2, dtype=np.int8)
    
    print(f"\n[MATRIX] Building {NUM_OBJECTS}×{NUM_ATTRIBUTES}...")
    
    matched = 0
    for obj_name, attr_dict in findings.items():
        if obj_name not in OBJECT_TO_IDX:
            continue
        
        obj_idx = OBJECT_TO_IDX[obj_name]
        
        for attr_name, value in attr_dict.items():
            if attr_name in ATTRIBUTE_TO_IDX:
                attr_idx = ATTRIBUTE_TO_IDX[attr_name]
                matrix[obj_idx, attr_idx] = value
                matched += 1
    
    non_unknown = np.sum(matrix != -2)
    print(f"[MATRIX] Populated {matched} cells")
    print(f"[MATRIX] Values: {np.sum(matrix == 1)} present, {np.sum(matrix == 0)} absent, "
          f"{np.sum(matrix == -1)} uncertain, {np.sum(matrix == -2)} unknown")
    
    state["scene_graph_matrix"] = matrix
    return {"scene_graph_matrix": matrix}


def aggregator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Final aggregation."""
    matrix = state.get("scene_graph_matrix")
    findings = state.get("verified_findings", {})
    
    if matrix is None:
        matrix = np.full((NUM_OBJECTS, NUM_ATTRIBUTES), -2, dtype=np.int8)
    
    metadata = {
        "objects": SG_OBJECTS,
        "attributes": SG_ATTRIBUTES,
        "attribute_categories": ATTRIBUTE_CATEGORIES,
        "matrix_shape": list(matrix.shape),
        "value_legend": {
            "1": "present",
            "0": "explicitly absent",
            "-1": "uncertain",
            "-2": "unknown/not mentioned"
        },
        "findings_summary": findings,
        "statistics": {
            "total_cells": int(matrix.size),
            "present": int(np.sum(matrix == 1)),
            "absent": int(np.sum(matrix == 0)),
            "uncertain": int(np.sum(matrix == -1)),
            "unknown": int(np.sum(matrix == -2)),
            "known_coverage": float(np.sum(matrix != -2) / matrix.size * 100)
        }
    }
    
    print(f"\n[FINAL] Matrix complete:")
    print(f"  +1 (present): {metadata['statistics']['present']}")
    print(f"  0 (absent): {metadata['statistics']['absent']}")
    print(f"  -1 (uncertain): {metadata['statistics']['uncertain']}")
    print(f"  -2 (unknown): {metadata['statistics']['unknown']}")
    print(f"  Known coverage: {metadata['statistics']['known_coverage']:.2f}%")
    
    state["final_matrix"] = matrix
    state["metadata"] = metadata
    
    return {"final_matrix": matrix, "metadata": metadata}