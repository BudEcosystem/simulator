#!/usr/bin/env python3
"""
Script to import hardware from hardware.json file into the database.
Handles the new data structure with instance types and vendor-specific pricing.
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hardware import BudHardware


def main():
    """Import hardware from hardware.json file."""
    # Default path to hardware.json
    hardware_json_path = Path(__file__).parent.parent / "src" / "utils" / "hardware.json"
    
    if len(sys.argv) > 1:
        hardware_json_path = Path(sys.argv[1])
    
    if not hardware_json_path.exists():
        print(f"Error: Hardware file not found at {hardware_json_path}")
        sys.exit(1)
    
    print(f"Importing hardware from: {hardware_json_path}")
    
    # Initialize BudHardware
    bud_hw = BudHardware()
    
    # Import from JSON
    result = bud_hw.import_from_json(str(hardware_json_path))
    
    # Print results
    print(f"\nImport Results:")
    print(f"  Imported: {result['imported']}")
    print(f"  Failed: {result['failed']}")
    
    if result['errors']:
        print(f"\nErrors:")
        for error in result['errors']:
            print(f"  - {error}")
    
    if result['success']:
        print("\nImport completed successfully!")
    else:
        print("\nImport completed with errors.")
        sys.exit(1)


if __name__ == "__main__":
    main() 