# agents/nodes.py - FIXED: Rule-based + LLM Hybrid Approach
import re
import os
import json
import numpy as np
from typing import Dict, Any, List, Tuple
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
    Values: 1 (present), -2 (explicitly absent), -1 (uncertain), 0 (not mentioned)
    """
    text_lower = text.lower()
    findings: Dict[str, int] = {}

    # Helper: detect negation/uncertainty around a match span
    def _negation_or_uncertain(ctx: str) -> Tuple[bool, bool]:
        # Returns (is_negated, is_uncertain)
        negation_tokens = [r'no', r'without', r'free of', r'clear of', r'not present', r'absent']
        uncertain_tokens = [r'possible', r'possibly', r'may represent', r'may be', r'suspicious for', r'concerning for', r'suggestive of']
        # Check within a window of up to 120 chars before the match
        window = ctx[-120:]
        for nt in negation_tokens:
            if re.search(rf"\b{nt}\b", window):
                return True, False
        for ut in uncertain_tokens:
            if re.search(rf"{ut}", window):
                return False, True
        return False, False

    # Build patterns for each known attribute (prefer SG_ATTRIBUTES canonical list)
    for attr in SG_ATTRIBUTES:
        attr_key = attr.lower().strip()
        # create a permissive regex for attribute (spaces -> \s+)
        pattern = re.escape(attr_key)
        pattern = pattern.replace(r'\ ', r'\s+')
        # Search all occurrences to capture negation context
        for m in re.finditer(rf'\b({pattern})\b', text_lower):
            span_start = max(0, m.start() - 120)
            ctx = text_lower[span_start:m.end()]
            is_neg, is_unc = _negation_or_uncertain(ctx)
            if is_unc:
                findings[attr_key] = -1
            elif is_neg:
                findings[attr_key] = -2
            else:
                findings[attr_key] = 1

    # Some additional heuristic patterns to capture common words not in SG_ATTRIBUTES
    heuristics = {
        'normal': [r'\bnormal\b', r'unremarkable', r'within normal limits'],
        'clear': [r'\bclear\b', r'well aerated'],
        'opacity': [r'opacit(?:y|ies)', r'densit(?:y|ies)'],
        'consolidation': [r'consolidation', r'consolidate'],
        'infiltrate': [r'infiltrat'],
        'pleural effusion': [r'pleural\s+effusion', r'\beffusion\b'],
        'pneumothorax': [r'pneumothorax'],
        'cardiomegaly': [r'cardiomegaly', r'cardiac\s+enlargement'],
        'atelectasis': [r'atelectasis'],
    }

    for attr, pats in heuristics.items():
        attr_key = attr.lower()
        if attr_key in findings:
            continue
        for pat in pats:
            for m in re.finditer(pat, text_lower):
                span_start = max(0, m.start() - 120)
                ctx = text_lower[span_start:m.end()]
                is_neg, is_unc = _negation_or_uncertain(ctx)
                if is_unc:
                    findings[attr_key] = -1
                elif is_neg:
                    findings[attr_key] = -2
                else:
                    findings[attr_key] = 1
                break
            if attr_key in findings:
                break

    # Object-specific tweaks
    obj_lower = object_name.lower()
    if 'cardiac' in obj_lower or 'cardio' in obj_lower:
        if findings.get('normal') == 1:
            findings['enlarged'] = 0
            findings['cardiomegaly'] = 0

    if 'lung' in obj_lower:
        if findings.get('clear') == 1:
            findings.setdefault('consolidation', 0)
            findings.setdefault('opacity', 0)

    # Normalize keys to simple form
    normalized: Dict[str, int] = {}
    for k, v in findings.items():
        key_clean = k.lower().strip()
        if v in [1, -1, -2]:
            normalized[key_clean] = int(v)

    return normalized


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
    - -2 = Explicitly absent (e.g., "no effusion", "no consolidation")
    - -1 = Uncertain (e.g., "suspicious for", "possible", "may represent")
    - DO NOT include 0 (0 will be used to mark attributes not mentioned at all)

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
                if isinstance(val, (int, float)) and val in [-1, -2, 1]:
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
    # Fallback: if no candidates found, create candidate entries for all SG_OBJECTS
    if not candidate_map:
        report_text = state.get("report_text", "")
        for obj in SG_OBJECTS:
            # try to capture sentences mentioning the object
            obj_lower = obj.lower()
            matched = [s for s in sentences if obj_lower in s.lower()]
            if matched:
                candidate_map[obj] = matched
            else:
                # As a last resort, include full report so rule-based methods can scan globally
                if report_text:
                    candidate_map[obj] = [report_text]

    # If LLM is unavailable, ensure we evaluate all objects so dataset is complete
    if not LLM_AVAILABLE:
        for obj in SG_OBJECTS:
            if obj not in candidate_map:
                report_text = state.get("report_text", "")
                candidate_map[obj] = [report_text] if report_text else []

    state["candidates"] = candidate_map
    return {"candidates": candidate_map}



def llm_enricher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: Actually call LLM and merge with rule-based findings.
    """
    candidates = state.get("candidates", {})
    if not candidates:
        return {"findings_dict": {}}

    findings_dict: Dict[str, Dict[str, int]] = {}

    for obj_name, phrases in candidates.items():
        # Ensure we have a string to analyze
        combined_text = " ".join([p for p in phrases if p]) if phrases else state.get("report_text", "")

        print(f"\n[PROCESS] {obj_name}")
        print(f"  Text: {combined_text[:100]}...")

        # 1. Rule-based extraction (baseline)
        rule_findings = extract_findings_rule_based(combined_text, obj_name)
        print(f"  Rule-based: {len(rule_findings)} findings")

        merged: Dict[str, int] = {k.lower().strip(): int(v) for k, v in rule_findings.items()}

        # 2. LLM extraction (enhancement) if available
        if LLM_AVAILABLE:
            llm_findings = extract_findings_llm(combined_text, obj_name, rule_findings)
            print(f"  LLM: {len(llm_findings)} findings")
            for attr, val in llm_findings.items():
                merged[attr.lower().strip()] = int(val)
        else:
            print("  LLM not available — using rule-based results only")

        clean_obj = obj_name.lower().strip()
        findings_dict[clean_obj] = merged
        print(f"  Final: {len(merged)} findings for {clean_obj}")

    state["findings_dict"] = findings_dict
    return {"findings_dict": findings_dict}


def llm_verifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: Actually verify findings using LLM.
    """
    findings = state.get("findings_dict", {})
    if not findings or not LLM_AVAILABLE:
        # Just validate values and ensure all SG_OBJECTS are present (so matrix will be full-dimension)
        verified: Dict[str, Dict[str, int]] = {}
        for obj in SG_OBJECTS:
            clean_obj = obj.lower().strip()
            attrs = findings.get(clean_obj) or findings.get(obj) or {}
            verified[clean_obj] = {k.lower().strip(): int(v) for k, v in attrs.items() if v in [-1, -2, 1]}
        state["verified_findings"] = verified
        return {"verified_findings": verified}
    
    print(f"\n[VERIFY] Validating {len(findings)} objects...")
    
    # Create verification prompt
    prompt = f"""You are a medical expert validating extracted radiology findings.

EXTRACTED FINDINGS:
{json.dumps(findings, indent=2)}

VALIDATION RULES:
1. Check logical consistency (e.g., can't have "normal" AND "cardiomegaly" both as 1)
4. Remove contradictions
2. Verify negations are correct (-2 = explicitly absent)
3. Check uncertainty markers (-1 = suspicious/possible)

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
                        if val in [-1, -2, 1]:
                            verified[clean_obj][clean_attr] = int(val)
                print(f"[VERIFY] ✓ Validated and normalized findings")
                state["verified_findings"] = verified
                return {"verified_findings": verified}
    except Exception as e:
        print(f"[VERIFY] Failed: {e}")
    
    # Fallback: basic validation
    verified = {}
    for obj, attrs in findings.items():
        verified[obj] = {k: v for k, v in attrs.items() if v in [-1, -2, 1]}
    
    state["verified_findings"] = verified
    return {"verified_findings": verified}


def matrix_builder_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build matrix with values: 0 (not mentioned), -2 (explicitly absent), -1 (uncertain), 1 (present).
    """
    findings = state.get("verified_findings", {})
    
    # Initialize with 0 (not mentioned)
    matrix = np.full((NUM_OBJECTS, NUM_ATTRIBUTES), 0, dtype=np.int8)
    
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
    
        non_unknown = np.sum(matrix != 0)
        print(f"[MATRIX] Populated {matched} cells")
        print(f"[MATRIX] Values: {np.sum(matrix == 1)} present, {np.sum(matrix == -2)} explicitly absent, "
            f"{np.sum(matrix == -1)} uncertain, {np.sum(matrix == 0)} not mentioned")
    
    state["scene_graph_matrix"] = matrix
    return {"scene_graph_matrix": matrix}


def aggregator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Final aggregation."""
    matrix = state.get("scene_graph_matrix")
    findings = state.get("verified_findings", {})
    
    if matrix is None:
        matrix = np.full((NUM_OBJECTS, NUM_ATTRIBUTES), 0, dtype=np.int8)
    
    metadata = {
        "objects": SG_OBJECTS,
        "attributes": SG_ATTRIBUTES,
        "attribute_categories": ATTRIBUTE_CATEGORIES,
        "matrix_shape": list(matrix.shape),
        "value_legend": {
            "1": "present",
            "0": "not mentioned",
            "-1": "uncertain",
            "-2": "explicitly absent"
        },
        "findings_summary": findings,
        "statistics": {
            "total_cells": int(matrix.size),
            "present": int(np.sum(matrix == 1)),
            "explicitly_absent": int(np.sum(matrix == -2)),
            "uncertain": int(np.sum(matrix == -1)),
            "not_mentioned": int(np.sum(matrix == 0)),
            "known_coverage": float(np.sum(matrix != 0) / matrix.size * 100)
        }
    }
    
    print(f"\n[FINAL] Matrix complete:")
    print(f"  +1 (present): {metadata['statistics']['present']}")
    print(f"  -2 (explicitly absent): {metadata['statistics']['explicitly_absent']}")
    print(f"  -1 (uncertain): {metadata['statistics']['uncertain']}")
    print(f"  0 (not mentioned): {metadata['statistics']['not_mentioned']}")
    print(f"  Known coverage: {metadata['statistics']['known_coverage']:.2f}%")
    
    state["final_matrix"] = matrix
    state["metadata"] = metadata
    
    return {"final_matrix": matrix, "metadata": metadata}