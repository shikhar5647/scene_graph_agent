# agents/nodes.py - SGRRG Schema Implementation
import re
import os
import json
import numpy as np
from typing import Dict, Any, List, Set
from utils.prompts import (    BASE_ENRICH_PROMPT, VERIFICATION_PROMPT)
from dotenv import load_dotenv
from utils.config import (
    SG_OBJECTS, SG_ATTRIBUTES, ATTRIBUTE_CATEGORIES,
    OBJECT_TO_IDX, ATTRIBUTE_TO_IDX,
    NUM_OBJECTS, NUM_ATTRIBUTES,
    ANATOMICAL_FINDINGS, DISEASE_FINDINGS, TUBES_LINES, DEVICES,
    TECHNICAL_ASSESSMENT, NLP_DESCRIPTORS, SEVERITY_DESCRIPTORS,
    TEMPORAL_DESCRIPTORS, TEXTURE_DESCRIPTORS
)

load_dotenv()

# Try to initialize Gemini
try:
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        client = genai.Client(api_key=api_key)
        LLM_AVAILABLE = True
        print("[INFO] Gemini client initialized")
    else:
        LLM_AVAILABLE = False
        print("[INFO] No API key - using rule-based extraction only")
except Exception:
    LLM_AVAILABLE = False
    print("[INFO] Using rule-based extraction only")


def _call_llm_safe(prompt: str) -> str:
    """Call LLM if available."""
    if not LLM_AVAILABLE:
        return ""
    try:
        if hasattr(client, "generate_content"):
            resp = client.generate_content(prompt)
            return getattr(resp, "text", str(resp))
        models = getattr(client, "models", None)
        if models and hasattr(models, "generate_content"):
            resp = models.generate_content(model="gemini-2.5-pro", contents=prompt)
            return getattr(resp, "text", str(resp))
    except Exception:
        pass
    return ""


# ============================================================================
# COMPREHENSIVE RULE-BASED EXTRACTION (SGRRG Schema)
# ============================================================================

def extract_findings_rule_based(text: str, object_name: str) -> Dict[str, int]:
    """
    Extract findings using comprehensive SGRRG attribute patterns.
    Returns: {attribute: value} where value ∈ {-1, 0, +1}
    """
    text_lower = text.lower()
    findings = {}
    
    # ========================================================================
    # PATTERN DEFINITIONS FOR ALL ATTRIBUTES
    # ========================================================================
    
    # POSITIVE patterns (attribute present = +1)
    positive_patterns = {
        # Anatomical
        'normal': [r'\bnormal\b', r'unremarkable', r'within normal limits'],
        'clear': [r'\bclear\b', r'well aerated'],
        'hyperinflated': [r'hyperinflat'],
        'hyperlucent': [r'hyperlucen'],
        'low lung volumes': [r'low\s+(?:lung\s+)?volumes'],
        'elevated': [r'elevated', r'elevation'],
        'flattened': [r'flattened'],
        'displaced': [r'displaced', r'displacement'],
        'enlarged': [r'enlarged', r'enlargement'],
        'tortuous': [r'tortuous'],
        'ectatic': [r'ectatic', r'ectasia'],
        'unfolded': [r'unfolded'],
        
        # Diseases
        'opacity': [r'opacit(?:y|ies)', r'infiltrat', r'densit(?:y|ies)'],
        'consolidation': [r'consolidation', r'consolidate'],
        'infiltrate': [r'infiltrat'],
        'atelectasis': [r'atelectasis', r'atelectatic'],
        'collapse': [r'\bcollapse\b', r'collapsed'],
        'pleural effusion': [r'pleural\s+effusion', r'\beffusion\b'],
        'pneumothorax': [r'pneumothorax', r'ptx'],
        'pneumomediastinum': [r'pneumomediastinum'],
        'subcutaneous emphysema': [r'subcutaneous\s+emphysema'],
        'pulmonary edema': [r'pulmonary\s+edema', r'\bedema\b'],
        'vascular congestion': [r'vascular\s+congestion', r'congestion'],
        'cardiomegaly': [r'cardiomegaly', r'cardiac\s+enlargement'],
        'mass': [r'\bmass\b', r'masses'],
        'nodule': [r'nodule', r'nodular'],
        'lesion': [r'lesion'],
        'granuloma': [r'granuloma'],
        'calcification': [r'calcification', r'calcified', r'calcific'],
        'fibrosis': [r'fibrosis', r'fibrotic'],
        'scarring': [r'scar(?:ring)?'],
        'thickening': [r'thickening', r'thickened'],
        'pleural thickening': [r'pleural\s+thickening'],
        'interstitial thickening': [r'interstitial(?:\s+thickening)?'],
        'blunted costophrenic angle': [r'blunted\s+(?:costophrenic\s+)?angle', r'blunting'],
        'pneumonia': [r'pneumonia'],
        'infection': [r'infection', r'infectious'],
        'fracture': [r'fracture'],
        'degenerative changes': [r'degenerative'],
        'hilar enlargement': [r'hilar\s+enlargement', r'enlarged\s+hil'],
        'lymphadenopathy': [r'lymphadenopathy', r'adenopathy'],
        'mediastinal widening': [r'mediastinal\s+widening', r'widened\s+mediastinum'],
        
        # Tubes/Lines
        'endotracheal tube': [r'endotracheal\s+tube', r'\bet\s+tube\b', r'\bett\b'],
        'nasogastric tube': [r'nasogastric\s+tube', r'\bng\s+tube\b', r'\bngt\b'],
        'chest tube': [r'chest\s+tube'],
        'central line': [r'central\s+(?:venous\s+)?(?:line|catheter)', r'\bcvc\b'],
        
        # Devices
        'pacemaker': [r'pacemaker'],
        'icd': [r'\bicd\b', r'aicd', r'defibrillator'],
        'prosthetic valve': [r'prosthetic\s+valve', r'valve\s+replacement'],
        'surgical clips': [r'surgical\s+clips', r'\bclips\b'],
        'sternotomy wires': [r'sternotomy\s+wires', r'sternal\s+wires'],
        
        # Technical
        'rotated': [r'rotated', r'rotation'],
        'lordotic': [r'lordotic'],
        'underpenetrated': [r'underpenetrated'],
        'overpenetrated': [r'overpenetrated'],
        
        # NLP descriptors
        'focal': [r'\bfocal\b'],
        'diffuse': [r'diffuse'],
        'patchy': [r'patchy'],
        'multifocal': [r'multifocal'],
        'bilateral': [r'bilateral', r'\bboth\b', r'bilaterally'],
        'unilateral': [r'unilateral'],
        'asymmetric': [r'asymmetric'],
        'symmetric': [r'symmetric'],
        'peripheral': [r'peripheral'],
        'central': [r'central'],
        'basilar': [r'basilar', r'\bbase\b', r'bases'],
        'apical': [r'apical', r'\bapex\b', r'apices'],
        'perihilar': [r'perihilar'],
        'retrocardiac': [r'retrocardiac'],
        
        # Severity
        'mild': [r'\bmild\b', r'mildly'],
        'moderate': [r'moderate', r'moderately'],
        'severe': [r'severe', r'severely'],
        'minimal': [r'minimal', r'minimally'],
        'small': [r'\bsmall\b'],
        'large': [r'\blarge\b'],
        'extensive': [r'extensive'],
        'marked': [r'marked', r'markedly'],
        'significant': [r'significant'],
        
        # Temporal
        'acute': [r'\bacute\b', r'acutely'],
        'chronic': [r'chronic', r'chronically'],
        'subacute': [r'subacute'],
        'new': [r'\bnew\b', r'newly'],
        'old': [r'\bold\b'],
        'stable': [r'stable'],
        'unchanged': [r'unchanged'],
        'improved': [r'improved', r'improving', r'improvement'],
        'worsened': [r'worsened', r'worsening'],
        'progressive': [r'progressive'],
        'resolving': [r'resolving'],
        'persistent': [r'persistent'],
        
        # Texture
        'hazy': [r'hazy'],
        'dense': [r'\bdense\b'],
        'ground glass': [r'ground\s+glass'],
        'reticular': [r'reticular'],
        'linear': [r'linear'],
        'streaky': [r'streaky'],
        'fluffy': [r'fluffy'],
        'confluent': [r'confluent'],
        'scattered': [r'scattered'],
    }
    
    # NEGATIVE patterns (explicitly absent = 0)
    negative_patterns = {
        'opacity': [r'no\s+opacit', r'clear\s+of\s+opacit', r'without\s+opacit'],
        'consolidation': [r'no\s+consolidation', r'clear\s+of\s+consolidation', r'without\s+consolidation'],
        'pleural effusion': [r'no\s+(?:pleural\s+)?effusion', r'without\s+effusion'],
        'pneumothorax': [r'no\s+pneumothorax'],
        'cardiomegaly': [r'no\s+cardiomegaly', r'normal\s+cardiac\s+size'],
        'enlarged': [r'not\s+enlarged', r'normal\s+(?:in\s+)?size'],
        'atelectasis': [r'no\s+atelectasis'],
        'mass': [r'no\s+mass'],
        'nodule': [r'no\s+nodule'],
        'fracture': [r'no\s+fracture'],
        'pulmonary edema': [r'no\s+(?:pulmonary\s+)?edema'],
        'infiltrate': [r'no\s+infiltrat'],
        'acute': [r'no\s+acute'],
    }
    
    # UNCERTAIN patterns (suspicious/possible = -1)
    uncertain_patterns = {
        'consolidation': [r'suspicious\s+for.*consolidation', r'possible.*consolidation', r'(?:may|might|could)\s+(?:be|represent).*consolidation'],
        'opacity': [r'suspicious\s+for.*opacit', r'possible.*opacit', r'questionable.*opacit'],
        'mass': [r'suspicious\s+(?:for\s+)?mass', r'possible\s+mass'],
        'pneumothorax': [r'possible\s+pneumothorax', r'questionable\s+pneumothorax'],
        'pneumonia': [r'suspicious\s+for.*pneumonia', r'possible.*pneumonia', r'concerning\s+for.*pneumonia'],
        'infection': [r'suspicious\s+for.*infection', r'concerning\s+for.*infection(?:ous)?'],
        'malignancy': [r'suspicious\s+for.*malignan', r'concerning\s+for.*malignan'],
        'pleural effusion': [r'possible.*effusion', r'questionable.*effusion'],
    }
    
    # ========================================================================
    # EXTRACTION LOGIC
    # ========================================================================
    
    # 1. Check UNCERTAIN first (highest priority)
    for attr, patterns in uncertain_patterns.items():
        if attr in ATTRIBUTE_TO_IDX:
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    findings[attr] = -1
                    break
    
    # 2. Check NEGATIVE patterns
    for attr, patterns in negative_patterns.items():
        if attr in ATTRIBUTE_TO_IDX and attr not in findings:
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    findings[attr] = 0
                    break
    
    # 3. Check POSITIVE patterns
    for attr, patterns in positive_patterns.items():
        if attr in ATTRIBUTE_TO_IDX and attr not in findings:
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    findings[attr] = 1
                    break
    
    # ========================================================================
    # OBJECT-SPECIFIC LOGIC
    # ========================================================================
    
    obj_lower = object_name.lower()
    
    # Cardiac objects
    if 'cardiac' in obj_lower or 'heart' in obj_lower:
        if findings.get('normal') == 1:
            findings['enlarged'] = 0
            findings['cardiomegaly'] = 0
    
    # Lung objects
    if 'lung' in obj_lower:
        if findings.get('clear') == 1:
            findings.setdefault('consolidation', 0)
            findings.setdefault('opacity', 0)
            findings.setdefault('infiltrate', 0)
    
    # Costophrenic angles - effusion relationship
    if 'costophrenic' in obj_lower:
        if findings.get('blunted costophrenic angle') == 1:
            findings['pleural effusion'] = -1  # Suggests but not confirms
    
    return findings


# ============================================================================
# PIPELINE NODES
# ============================================================================

def split_report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Split report into sentences."""
    report: str = state.get("report_text", "").strip()
    if not report:
        return {"error": "no report_text provided"}
    
    sentences = []
    for line in report.splitlines():
        line = line.strip()
        if not line or line.startswith('Exam:'):
            continue
        # Split on periods
        parts = re.split(r'(?<!\d)\.(?!\d)\s+', line)
        for p in parts:
            p = p.strip()
            if p and len(p) > 5:
                sentences.append(p)
    
    print(f"[DEBUG] Split into {len(sentences)} sentences")
    state["sentences"] = sentences
    return {"sentences": sentences}


def candidate_extractor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract object mentions with comprehensive pattern matching."""
    sentences: List[str] = state.get("sentences", [])
    candidate_map = {}
    
    # Object detection patterns
    object_patterns = {
        "cardiac silhouette": [r'cardiac(?:\s+silhouette)?', r'\bheart\b', r'cardiomegaly'],
        "left lung": [r'left\s+lung', r'left\s+hemithorax', r'on\s+the\s+left(?!\s+(?:lower|mid|upper))'],
        "right lung": [r'right\s+lung', r'right\s+hemithorax', r'on\s+the\s+right(?!\s+(?:lower|mid|upper))'],
        "left lower lung zone": [r'left\s+lower\s+(?:lung\s+)?(?:zone|lobe)', r'left\s+base', r'left\s+basilar'],
        "right lower lung zone": [r'right\s+lower\s+(?:lung\s+)?(?:zone|lobe)', r'right\s+base', r'right\s+basilar'],
        "left mid lung zone": [r'left\s+mid(?:dle)?\s+(?:lung\s+)?zone', r'left\s+mid\s+lobe'],
        "right mid lung zone": [r'right\s+mid(?:dle)?\s+(?:lung\s+)?zone', r'right\s+mid\s+lobe'],
        "left upper lung zone": [r'left\s+upper\s+(?:lung\s+)?(?:zone|lobe)', r'left\s+apex', r'left\s+apical'],
        "right upper lung zone": [r'right\s+upper\s+(?:lung\s+)?(?:zone|lobe)', r'right\s+apex', r'right\s+apical'],
        "left apical zone": [r'left\s+apical?\s+zone', r'left\s+apex'],
        "right apical zone": [r'right\s+apical?\s+zone', r'right\s+apex'],
        "left costophrenic angle": [r'left\s+costophrenic', r'left\s+cp\s+angle'],
        "right costophrenic angle": [r'right\s+costophrenic', r'right\s+cp\s+angle'],
        "left hemidiaphragm": [r'left\s+(?:hemi)?diaphragm'],
        "right hemidiaphragm": [r'right\s+(?:hemi)?diaphragm'],
        "mediastinum": [r'mediastin'],
        "upper mediastinum": [r'upper\s+mediastin', r'superior\s+mediastin'],
        "left hilar structures": [r'left\s+hil(?:ar|um)'],
        "right hilar structures": [r'right\s+hil(?:ar|um)'],
        "aortic arch": [r'aortic\s+arch', r'\baorta\b'],
        "trachea": [r'trachea'],
        "carina": [r'carina'],
        "spine": [r'spine', r'vertebra', r'osseous'],
        "left clavicle": [r'left\s+clavicle'],
        "right clavicle": [r'right\s+clavicle'],
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
        
        # If "lungs" mentioned without side, add to both
        if re.search(r'\blungs\b', s_lower) and not re.search(r'left|right', s_lower):
            for obj in ["left lung", "right lung"]:
                if obj not in candidate_map:
                    candidate_map[obj] = []
                if s not in candidate_map[obj]:
                    candidate_map[obj].append(s)
    
    print(f"\n[DEBUG] Found {len(candidate_map)} objects with mentions")
    state["candidates"] = candidate_map
    return {"candidates": candidate_map}


def llm_enricher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract findings using rule-based + optional LLM enhancement."""
    candidates: Dict[str, List[str]] = state.get("candidates", {})
    if not candidates:
        return {"findings_dict": {}}
    
    findings_dict = {}
    
    for obj_name, phrases in candidates.items():
        combined_text = " ".join(phrases)
        
        # Always use rule-based
        rule_findings = extract_findings_rule_based(combined_text, obj_name)
        
        print(f"\n[{obj_name}]")
        print(f"  Sentences: {len(phrases)}")
        print(f"  Rule findings: {len(rule_findings)} attributes")
        
        findings_dict[obj_name] = rule_findings
    
    state["findings_dict"] = findings_dict
    return {"findings_dict": findings_dict}


def llm_verifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Verify findings are valid."""
    findings = state.get("findings_dict", {})
    
    verified = {}
    for obj_name, attr_dict in findings.items():
        verified_attrs = {k: v for k, v in attr_dict.items() if v in [-1, 0, 1]}
        if verified_attrs:
            verified[obj_name] = verified_attrs
    
    state["verified_findings"] = verified
    return {"verified_findings": verified}


def matrix_builder_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build the 29×180 SGRRG matrix."""
    findings: Dict[str, Dict[str, int]] = state.get("verified_findings", {})
    
    matrix = np.zeros((NUM_OBJECTS, NUM_ATTRIBUTES), dtype=np.int8)
    
    print(f"\n[DEBUG] Building {NUM_OBJECTS}×{NUM_ATTRIBUTES} matrix...")
    
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
    
    print(f"  Populated: {matched} cells")
    print(f"  Non-zero: {np.count_nonzero(matrix)}")
    
    state["scene_graph_matrix"] = matrix
    return {"scene_graph_matrix": matrix}


def aggregator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Final aggregation with statistics."""
    matrix = state.get("scene_graph_matrix")
    findings = state.get("verified_findings", {})
    
    if matrix is None:
        matrix = np.zeros((NUM_OBJECTS, NUM_ATTRIBUTES), dtype=np.int8)
    
    metadata = {
        "objects": SG_OBJECTS,
        "attributes": SG_ATTRIBUTES,
        "attribute_categories": ATTRIBUTE_CATEGORIES,
        "matrix_shape": list(matrix.shape),
        "value_legend": {"+1": "present", "0": "absent", "-1": "uncertain"},
        "findings_summary": findings,
        "statistics": {
            "total_cells": int(matrix.size),
            "positive": int(np.sum(matrix == 1)),
            "negative": int(np.sum(matrix == 0)),
            "uncertain": int(np.sum(matrix == -1)),
            "non_zero": int(np.count_nonzero(matrix)),
            "coverage": float(np.sum(matrix != 0) / matrix.size * 100)
        }
    }
    
    print(f"\n[FINAL] Matrix: {matrix.shape}")
    print(f"  +1: {metadata['statistics']['positive']}")
    print(f"  0: {metadata['statistics']['negative']}")
    print(f"  -1: {metadata['statistics']['uncertain']}")
    print(f"  Coverage: {metadata['statistics']['coverage']:.2f}%")
    
    state["final_matrix"] = matrix
    state["metadata"] = metadata
    
    return {"final_matrix": matrix, "metadata": metadata}