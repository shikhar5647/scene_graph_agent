# utils/config.py
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

# Category IDs commonly used in Chest ImaGenome / SGRRG
CATEGORY_IDS = [
    "anatomicalfinding",
    "disease",
    "nlp",
    "technicalassessment",
    "tubesandlines",
    "devices",
]

# Comprehensive attribute list based on SGRRG and Chest ImaGenome
# These are the most common radiological findings/attributes
SG_ATTRIBUTES = [
    # Anatomical/Size findings
    "normal",
    "enlarged",
    "cardiomegaly",
    "tortuous",
    "ectatic",
    "displaced",
    "shifted",
    "elevated",
    "flattened",
    
    # Lung parenchyma findings
    "clear",
    "opacity",
    "consolidation",
    "infiltrate",
    "nodule",
    "mass",
    "atelectasis",
    "collapse",
    "hyperinflation",
    "hyperlucency",
    "ground glass opacity",
    "interstitial thickening",
    "fibrosis",
    "scarring",
    "granuloma",
    "calcification",
    "cavity",
    "cyst",
    
    # Pleural findings
    "pleural effusion",
    "pneumothorax",
    "pleural thickening",
    "pleural calcification",
    "blunted costophrenic angle",
    
    # Vascular findings
    "pulmonary edema",
    "vascular congestion",
    "pulmonary hypertension",
    "vascular calcification",
    
    # Mediastinal findings
    "mediastinal widening",
    "hilar enlargement",
    "lymphadenopathy",
    
    # Bony findings
    "fracture",
    "degenerative changes",
    "lytic lesion",
    "sclerotic lesion",
    "osseous abnormality",
    
    # Devices/Lines
    "endotracheal tube",
    "nasogastric tube",
    "central line",
    "pacemaker",
    "icd",
    "chest tube",
    "surgical clips",
    "prosthetic valve",
    
    # Temporal/Quality descriptors
    "acute",
    "chronic",
    "unchanged",
    "improved",
    "worsened",
    "new",
    "old",
    
    # Severity descriptors
    "mild",
    "moderate",
    "severe",
    "minimal",
    "extensive",
    
    # Distribution patterns
    "diffuse",
    "focal",
    "patchy",
    "bilateral",
    "unilateral",
    "multifocal",
    
    # Special findings
    "air bronchogram",
    "silhouette sign",
    "free air",
    "subcutaneous emphysema",
]

# Create index mappings for matrix construction
OBJECT_TO_IDX = {obj: idx for idx, obj in enumerate(SG_OBJECTS)}
ATTRIBUTE_TO_IDX = {attr: idx for idx, attr in enumerate(SG_ATTRIBUTES)}
IDX_TO_OBJECT = {idx: obj for idx, obj in enumerate(SG_OBJECTS)}
IDX_TO_ATTRIBUTE = {idx: attr for idx, attr in enumerate(SG_ATTRIBUTES)}

# Matrix dimensions
NUM_OBJECTS = len(SG_OBJECTS)
NUM_ATTRIBUTES = len(SG_ATTRIBUTES)