# utils/config.py - Complete SGRRG/Chest ImaGenome Schema

# 29 Anatomical Objects (Bounding Boxes)
SG_OBJECTS = [
    "left lung",
    "right lung",
    "cardiac silhouette",
    "mediastinum",
    "left lower lung zone",
    "right lower lung zone",
    "right hilar structures",
    "left hilar structures",
    "upper mediastinum",
    "left costophrenic angle",
    "right costophrenic angle",
    "left mid lung zone",
    "right mid lung zone",
    "aortic arch",
    "right upper lung zone",
    "left upper lung zone",
    "right hemidiaphragm",
    "right clavicle",
    "left clavicle",
    "left hemidiaphragm",
    "right apical zone",
    "trachea",
    "left apical zone",
    "carina",
    "svc",
    "right atrium",
    "cavoatrial junction",
    "abdomen",
    "spine",
]

# Complete Attribute Categories from SGRRG Paper
# Based on Chest ImaGenome scene graph annotations

# Category 1: Anatomical Findings (anatomicalfinding)
ANATOMICAL_FINDINGS = [
    "normal",
    "clear",
    "unremarkable",
    "hyperinflated",
    "hyperlucent",
    "low lung volumes",
    "elevated",
    "flattened",
    "displaced",
    "enlarged",
    "tortuous",
    "ectatic",
    "unfolded",
]

# Category 2: Diseases/Pathology (disease)
DISEASE_FINDINGS = [
    "opacity",
    "consolidation",
    "infiltrate",
    "atelectasis",
    "collapse",
    "pleural effusion",
    "pneumothorax",
    "pneumomediastinum",
    "subcutaneous emphysema",
    "pulmonary edema",
    "vascular congestion",
    "cardiomegaly",
    "mass",
    "nodule",
    "lesion",
    "granuloma",
    "calcification",
    "fibrosis",
    "scarring",
    "thickening",
    "pleural thickening",
    "interstitial thickening",
    "blunted costophrenic angle",
    "pneumonia",
    "infection",
    "tuberculosis",
    "malignancy",
    "metastasis",
    "fracture",
    "dislocation",
    "degenerative changes",
    "scoliosis",
    "kyphosis",
    "osteopenia",
    "lytic lesion",
    "sclerotic lesion",
    "hilar enlargement",
    "lymphadenopathy",
    "mediastinal widening",
    "aortic dissection",
    "aneurysm",
    "hernia",
    "diaphragmatic hernia",
]

# Category 3: Tubes and Lines (tubesandlines)
TUBES_LINES = [
    "endotracheal tube",
    "tracheostomy tube",
    "nasogastric tube",
    "orogastric tube",
    "chest tube",
    "pigtail catheter",
    "central venous catheter",
    "central line",
    "picc line",
    "swan-ganz catheter",
    "dialysis catheter",
]

# Category 4: Devices (devices)
DEVICES = [
    "pacemaker",
    "pacemaker lead",
    "icd",
    "aicd",
    "prosthetic valve",
    "valve replacement",
    "cabg",
    "surgical clips",
    "surgical staples",
    "sternotomy wires",
    "stent",
    "filter",
    "coil",
]

# Category 5: Technical Assessment (technicalassessment)
TECHNICAL_ASSESSMENT = [
    "rotated",
    "low lung volumes",
    "lordotic",
    "underpenetrated",
    "overpenetrated",
    "motion artifact",
    "poor inspiration",
    "adequate inspiration",
    "good positioning",
    "portable technique",
    "ap technique",
    "pa technique",
]

# Category 6: NLP-derived descriptors (nlp)
NLP_DESCRIPTORS = [
    "focal",
    "diffuse",
    "patchy",
    "multifocal",
    "bilateral",
    "unilateral",
    "asymmetric",
    "symmetric",
    "peripheral",
    "central",
    "basilar",
    "apical",
    "perihilar",
    "retrocardiac",
    "lingular",
]

# Category 7: Severity/Temporal/Texture Modifiers
SEVERITY_DESCRIPTORS = [
    "mild",
    "moderate",
    "severe",
    "minimal",
    "small",
    "large",
    "extensive",
    "marked",
    "significant",
]

TEMPORAL_DESCRIPTORS = [
    "acute",
    "chronic",
    "subacute",
    "new",
    "old",
    "stable",
    "unchanged",
    "improved",
    "worsened",
    "progressive",
    "resolving",
    "persistent",
]

TEXTURE_DESCRIPTORS = [
    "hazy",
    "dense",
    "ground glass",
    "reticular",
    "nodular",
    "linear",
    "streaky",
    "fluffy",
    "confluent",
    "scattered",
]

# Combine all attributes
SG_ATTRIBUTES = (
    ANATOMICAL_FINDINGS +
    DISEASE_FINDINGS +
    TUBES_LINES +
    DEVICES +
    TECHNICAL_ASSESSMENT +
    NLP_DESCRIPTORS +
    SEVERITY_DESCRIPTORS +
    TEMPORAL_DESCRIPTORS +
    TEXTURE_DESCRIPTORS
)

# Category mapping for each attribute
ATTRIBUTE_CATEGORIES = {}
for attr in ANATOMICAL_FINDINGS:
    ATTRIBUTE_CATEGORIES[attr] = "anatomicalfinding"
for attr in DISEASE_FINDINGS:
    ATTRIBUTE_CATEGORIES[attr] = "disease"
for attr in TUBES_LINES:
    ATTRIBUTE_CATEGORIES[attr] = "tubesandlines"
for attr in DEVICES:
    ATTRIBUTE_CATEGORIES[attr] = "devices"
for attr in TECHNICAL_ASSESSMENT:
    ATTRIBUTE_CATEGORIES[attr] = "technicalassessment"
for attr in NLP_DESCRIPTORS:
    ATTRIBUTE_CATEGORIES[attr] = "nlp"
for attr in SEVERITY_DESCRIPTORS:
    ATTRIBUTE_CATEGORIES[attr] = "severity"
for attr in TEMPORAL_DESCRIPTORS:
    ATTRIBUTE_CATEGORIES[attr] = "temporal"
for attr in TEXTURE_DESCRIPTORS:
    ATTRIBUTE_CATEGORIES[attr] = "texture"

# Create index mappings
OBJECT_TO_IDX = {obj: idx for idx, obj in enumerate(SG_OBJECTS)}
ATTRIBUTE_TO_IDX = {attr: idx for idx, attr in enumerate(SG_ATTRIBUTES)}
IDX_TO_OBJECT = {idx: obj for idx, obj in enumerate(SG_OBJECTS)}
IDX_TO_ATTRIBUTE = {idx: attr for idx, attr in enumerate(SG_ATTRIBUTES)}

# Matrix dimensions
NUM_OBJECTS = len(SG_OBJECTS)  # 29
NUM_ATTRIBUTES = len(SG_ATTRIBUTES)  # ~180

# For reference: Category IDs used in SGRRG
CATEGORY_IDS = [
    "anatomicalfinding",
    "disease",
    "tubesandlines",
    "devices",
    "technicalassessment",
    "nlp",
    "severity",
    "temporal",
    "texture",
]

print(f"[CONFIG] Loaded SGRRG schema: {NUM_OBJECTS} objects × {NUM_ATTRIBUTES} attributes")