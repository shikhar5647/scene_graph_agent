# utils/prompts.py
from .config import SG_OBJECTS, SG_ATTRIBUTES, CATEGORY_IDS

OBJ_LIST_STR = "\n".join(f"- {o}" for o in SG_OBJECTS)
ATTR_LIST_STR = "\n".join(f"- {a}" for a in SG_ATTRIBUTES)

BASE_ENRICH_PROMPT = f"""
You are a clinical NLP assistant that extracts structured radiology findings from chest X-ray reports.

Your task is to analyze report sentences and determine which attributes apply to which anatomical objects.

ANATOMICAL OBJECTS (29 total):
{OBJ_LIST_STR}

ATTRIBUTES TO CONSIDER:
{ATTR_LIST_STR}

For each object mentioned in the report, determine which attributes apply:
- **Present (+1)**: The attribute is explicitly stated or clearly implied to be present for this object
- **Absent (0)**: The attribute is explicitly stated to be absent (e.g., "no pleural effusion")
- **Uncertain (-1)**: The report suggests uncertainty about the attribute (e.g., "possible", "suspicious for", "cannot exclude")

IMPORTANT RULES:
1. Only assign attributes that are mentioned or clearly implied in the report text
2. For attributes not mentioned at all for an object, leave them out (they will default to 0)
3. Pay attention to negations: "no consolidation" = 0 (absent)
4. Pay attention to uncertainty: "suspicious for", "possible", "concerning for" = -1 (uncertain)
5. Normal/clear findings should be marked as +1 for "normal" or "clear" attributes
6. Consider anatomical relationships (e.g., if "right lower lobe opacity" is mentioned, it affects "right lower lung zone")

Output format: JSON object with structure:
{{
  "bbox_name": {{
    "attribute_name": +1 or 0 or -1,
    "attribute_name2": +1 or 0 or -1,
    ...
  }}
}}

Only include objects and attributes explicitly mentioned or negated in the report.
"""

VERIFICATION_PROMPT = """
You are validating extracted radiology findings for consistency and accuracy.

Review the extracted findings JSON and:
1. Ensure all attribute values are +1 (present), 0 (absent), or -1 (uncertain)
2. Check for logical consistency (e.g., can't have both "normal" and "consolidation" as +1)
3. Verify negations are properly captured as 0
4. Ensure uncertainty markers are captured as -1
5. Remove any duplicate or contradictory entries

Return the corrected JSON in the same format.
"""