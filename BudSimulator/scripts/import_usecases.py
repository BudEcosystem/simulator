#!/usr/bin/env python3
"""
Script to import usecases from JSON file into the database.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.usecases import BudUsecases


def main():
    """Import usecases from the default JSON file."""
    # Initialize usecase manager
    usecase_manager = BudUsecases()
    
    # Path to usecases JSON file
    json_file = Path(__file__).parent.parent / "src" / "utils" / "usecases.json"
    
    if not json_file.exists():
        print(f"Error: Usecases file not found at {json_file}")
        return 1
    
    print(f"Importing usecases from {json_file}")
    
    try:
        # Import usecases
        stats = usecase_manager.import_from_json(str(json_file))
        
        print("\nImport Statistics:")
        print(f"  Total usecases: {stats['total']}")
        print(f"  Successfully imported: {stats['imported']}")
        print(f"  Skipped (already exists): {stats['skipped']}")
        print(f"  Errors: {len(stats['errors'])}")
        
        if stats['errors']:
            print("\nErrors encountered:")
            for error in stats['errors']:
                print(f"  - {error['usecase']}: {error['error']}")
        
        # Show summary statistics
        print("\nDatabase Statistics:")
        db_stats = usecase_manager.get_stats()
        print(f"  Total usecases in database: {db_stats['total_usecases']}")
        print(f"\n  By Industry:")
        for industry, count in sorted(db_stats['by_industry'].items()):
            print(f"    - {industry}: {count}")
        print(f"\n  By Latency Profile:")
        for profile, count in sorted(db_stats['by_latency_profile'].items()):
            print(f"    - {profile}: {count}")
        
        return 0
        
    except Exception as e:
        print(f"Error during import: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 