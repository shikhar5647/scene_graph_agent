"""
Visualize the generated scene graph matrix.
Usage: python visualize_matrix.py
"""

import numpy as np
import json
from pathlib import Path

def visualize_matrix():
    """Load and visualize the generated matrix."""
    
    # Load data
    if not Path("sgrrg_matrix.npy").exists():
        print("❌ No matrix found. Run quick_test.py first!")
        return
    
    matrix = np.load("sgrrg_matrix.npy")
    
    with open("sgrrg_metadata.json", "r") as f:
        metadata = json.load(f)
    
    objects = metadata['objects']
    attributes = metadata['attributes']
    attr_cats = metadata['attribute_categories']
    
    print("="*80)
    print(f"SGRRG MATRIX VISUALIZATION - {matrix.shape[0]}×{matrix.shape[1]}")
    print("="*80)
    
    # Statistics
    stats = metadata['statistics']
    print(f"\n📊 Overall Statistics:")
    print(f"   Present (+1): {stats['positive']}")
    print(f"   Absent (0): {stats['negative']}")
    print(f"   Uncertain (-1): {stats['uncertain']}")
    print(f"   Coverage: {stats['coverage']:.2f}%")
    
    # Per-object summary
    print(f"\n📋 Per-Object Summary:")
    print("-"*80)
    
    for i, obj in enumerate(objects):
        row = matrix[i, :]
        n_findings = np.count_nonzero(row)
        
        if n_findings > 0:
            n_pos = np.sum(row == 1)
            n_neg = np.sum(row == 0)
            n_unc = np.sum(row == -1)
            
            print(f"\n{obj}:")
            print(f"   Total findings: {n_findings}")
            print(f"   +1: {n_pos}, 0: {n_neg}, -1: {n_unc}")
            
            # List attributes
            attrs_found = []
            for j in range(matrix.shape[1]):
                if row[j] != 0:
                    val_sym = {1: "✓", 0: "✗", -1: "⚠"}.get(row[j])
                    cat = attr_cats.get(attributes[j], "?")
                    attrs_found.append(f"{val_sym} {attributes[j]} [{cat}]")
            
            for attr_str in attrs_found:
                print(f"      {attr_str}")
    
    # Category breakdown
    print(f"\n\n📊 Findings by Category:")
    print("-"*80)
    
    category_counts = {}
    for j, attr in enumerate(attributes):
        cat = attr_cats.get(attr, "unknown")
        col = matrix[:, j]
        
        if np.count_nonzero(col) > 0:
            if cat not in category_counts:
                category_counts[cat] = {"present": 0, "absent": 0, "uncertain": 0}
            
            category_counts[cat]["present"] += int(np.sum(col == 1))
            category_counts[cat]["absent"] += int(np.sum(col == 0))
            category_counts[cat]["uncertain"] += int(np.sum(col == -1))
    
    for cat, counts in sorted(category_counts.items()):
        total = sum(counts.values())
        print(f"\n{cat}:")
        print(f"   Total: {total} | +1: {counts['present']} | 0: {counts['absent']} | -1: {counts['uncertain']}")
    
    # Compact matrix view (show only non-zero)
    print(f"\n\n📊 Compact Matrix View (non-zero entries only):")
    print("-"*80)
    print(f"{'Object':<30} | {'Attribute':<30} | Value | Category")
    print("-"*80)
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                val_str = {1: "+1 ✓", 0: " 0 ✗", -1: "-1 ⚠"}.get(matrix[i, j])
                cat = attr_cats.get(attributes[j], "?")
                print(f"{objects[i]:<30} | {attributes[j]:<30} | {val_str} | {cat}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    visualize_matrix()