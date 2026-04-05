#!/usr/bin/env python
"""Run the parser with OCR debug mode to capture OCR input images"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fiadoc.parser import QualifyingParser
from fiadoc.parser import page as page_module

# Enable OCR debug mode
debug_dir = Path('debug_ocr')
page_module.OCR_DEBUG_DIR = str(debug_dir)

print("OCR Debug Mode: Enabled")
print(f"Debug directory: {debug_dir.absolute()}")
print(f"OCR images will be saved to: {debug_dir.absolute()}\n")

# Run the parser
print("=" * 72)
print("Running RaceParser with OCR debug mode...\n")

parser = QualifyingParser(
    classification_file='data/pdf/2026_1_quali_final_classification.pdf',
    lap_times_file=None,
    year=2026,
    round_no=1,
    session='quali'
)

try:
    _ = parser._parse_classification()
    print('Parsing completed successfully')
except Exception as e:
    print(f'Error: {e}\n')
    raise
