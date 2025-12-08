#!/usr/bin/env python3
"""Batch runner for SGRRG pipeline.

Reads an input CSV (default: `mimic_label_report.csv`), finds the report text and the image/file path
for each row, runs the pipeline `run_graph(report_text)` and writes a per-report CSV containing the
29×N attribute matrix. Output CSV filenames are derived from the image filepath so that
`.../files/p14/p14692813/s54911261/<uuid>.png` -> `p14692813/s54911261/<uuid>.csv`.

Usage:
    python scripts/run_batch_pipeline.py \
        --input mimic_label_report.csv \
        --output-dir output_matrices \
        --file-col image_path --report-col report

The script will create subdirectories under `--output-dir` matching the relative paths
(`p14692813/s54911261/...`) and save the CSVs there. Metadata JSON is also saved alongside.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from agents.graph import run_graph


def find_columns(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first column name in df that matches any candidate substrings (case-insensitive)."""
    cols = list(df.columns)
    lower = [c.lower() for c in cols]
    for cand in candidates:
        for i, c in enumerate(lower):
            if cand.lower() in c:
                return cols[i]
    return None


def make_relative_csv_name(orig_path: str) -> str:
    """Derive the relative CSV name from an original image path.

    Example:
      /workspace/.../files/p14/p14692813/s54911261/8dcfd56d-...png
    -> p14692813/s54911261/8dcfd56d-....csv
    """
    if not isinstance(orig_path, str) or not orig_path:
        raise ValueError("Empty path")

    p = orig_path.replace('\\', '/')
    # Look for '/files/' as anchor
    if '/files/' in p:
        after = p.split('/files/', 1)[1]
        parts = after.split('/')
        # If first part is short patient folder like 'p14', drop it
        if len(parts) > 1 and parts[0].startswith('p') and len(parts[0]) <= 4:
            parts = parts[1:]
        # Replace extension with .csv
        parts[-1] = Path(parts[-1]).with_suffix('.csv').name
        return '/'.join(parts)

    # Fallback: use last three components if available
    parts = p.split('/')
    tail = parts[-3:] if len(parts) >= 3 else parts
    tail[-1] = Path(tail[-1]).with_suffix('.csv').name
    return '/'.join(tail)


def run_for_row(report_text: str, out_csv_path: Path) -> dict:
    """Run the pipeline for a single report and write the matrix CSV and metadata JSON.

    Returns the metadata dict from the pipeline.
    """
    result = run_graph(report_text)

    matrix = result.get('matrix')
    metadata = result.get('metadata', {})

    # Ensure matrix and metadata present
    if matrix is None:
        raise RuntimeError('Pipeline did not return a matrix for this report')

    objects = metadata.get('objects') or list(range(matrix.shape[0]))
    attributes = metadata.get('attributes') or list(range(matrix.shape[1]))

    df = pd.DataFrame(matrix, index=objects, columns=attributes)

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=True)

    # Save metadata next to CSV
    meta_path = out_csv_path.with_suffix('.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata


def main():
    parser = argparse.ArgumentParser(description='Batch run SGRRG pipeline over a CSV of reports')
    parser.add_argument('--input', '-i', default='mimic_label_report.csv', help='Input CSV file with report text and image path')
    parser.add_argument('--output-dir', '-o', default='output_matrices', help='Directory to write per-report CSVs')
    parser.add_argument('--file-col', help='Column name containing image/file path (optional)')
    parser.add_argument('--report-col', default='report', help="Column name containing report text (default: 'report'). Will fall back to auto-detection if not present.")
    parser.add_argument('--skip-existing', action='store_true', help='Skip reports with existing output CSV')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of rows processed (0 = all)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    df = pd.read_csv(args.input)
    logging.info('Loaded input CSV with %d rows', len(df))

    # Determine report column: prefer explicit/default value, but fall back to auto-detection
    report_col = args.report_col
    if report_col not in df.columns:
        detected = find_columns(df, ['report', 'findings', 'impression', 'note', 'text'])
        if detected:
            logging.info("Requested report column '%s' not found; using detected column '%s'", report_col, detected)
            report_col = detected
        else:
            raise RuntimeError("Could not find a report text column. Provide --report-col with the correct column name.")

    # Determine file path column: prefer explicit value if given, else try auto-detection
    file_col = args.file_col
    if file_col and file_col not in df.columns:
        logging.warning("Requested file column '%s' not found; attempting auto-detection", file_col)
        file_col = None

    if not file_col:
        file_col = find_columns(df, ['file', 'path', 'filename', 'image'])
        if not file_col:
            logging.warning('No file path column detected; output filenames will be based on row index')

    out_root = Path(args.output_dir)

    total = len(df) if args.limit <= 0 else min(len(df), args.limit)
    logging.info('Processing %d reports', total)

    for idx, row in df.iterrows():
        if args.limit and idx >= args.limit:
            break

        try:
            report_text = str(row.get(report_col, '')).strip() if report_col else ''
            if not report_text:
                logging.info('Row %d: empty report, skipping', idx)
                continue

            orig_path = str(row.get(file_col, '')) if file_col else ''
            try:
                rel_name = make_relative_csv_name(orig_path) if orig_path else f'row_{idx}.csv'
            except Exception:
                rel_name = f'row_{idx}.csv'

            out_path = out_root / rel_name

            if args.skip_existing and out_path.exists():
                logging.info('Row %d: %s exists, skipping', idx, out_path)
                continue

            logging.info('Row %d: processing -> %s', idx, out_path)

            metadata = run_for_row(report_text, out_path)
            logging.info('Row %d: done (present=%s, uncertain=%s, explicitly_absent=%s)',
                         idx,
                         metadata.get('statistics', {}).get('present'),
                         metadata.get('statistics', {}).get('uncertain'),
                         metadata.get('statistics', {}).get('explicitly_absent'))

        except Exception as e:
            logging.exception('Row %d: failed with error: %s', idx, e)


if __name__ == '__main__':
    main()
