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

For each object mentioned in the report, determine which attributes apply.

Encoding (numeric):
- `1` = present / affirmed
- `-1` = uncertain / possible
- `-2` = explicitly absent (negated)
- `0` = not mentioned for that object (will be used as the sparse default in the matrix)

IMPORTANT RULES:
1. Only assign attributes that are mentioned or clearly implied in the report text.
2. For attributes not mentioned at all for an object, do NOT include them in the object's dictionary; they will be encoded as `0` in the final matrix.
3. Use `-2` to indicate an attribute is explicitly absent (negated) in the text (e.g., "no effusion").
4. Use `-1` for uncertainty cues (e.g., "possible", "suspicious for").
5. Use `1` for clear affirmations (e.g., "consolidation", "opacity").
6. Keep attribute labels short and normalized (lowercase).
7. Consider anatomical relationships (e.g., if "right lower lobe opacity" is mentioned, it should affect "right lower lung zone").

Output format: JSON object with structure (only include mentioned/negated attributes):
```
{
  "bbox_name": {
    "attribute_name": 1,
    "attribute_name2": -2,
    "attribute_name3": -1
  }
}
```

Only output VALID JSON that follows the schema above.
"""

VERIFICATION_PROMPT = """
You are validating extracted radiology findings for consistency and accuracy.

Review the extracted findings JSON and:
1. Ensure all attribute values use the encoding: 1 (present), -1 (uncertain), -2 (explicitly absent), 0 (not mentioned - should not appear explicitly in the JSON)
2. Check for logical consistency (e.g., can't have both "normal" and "consolidation" as 1)
3. Verify negations are properly captured as -2
4. Ensure uncertainty markers are captured as -1
5. Remove any duplicate or contradictory entries

Return the corrected JSON in the same format. Output ONLY valid JSON.
"""