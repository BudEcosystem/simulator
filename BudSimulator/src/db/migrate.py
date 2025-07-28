"""
Database migration script to add logo and model_analysis columns.
"""

import sqlite3
import logging
from pathlib import Path
from .connection import DatabaseConnection

logger = logging.getLogger(__name__)

def migrate_add_logo_and_analysis():
    """Add logo and model_analysis columns to the models table."""
    
    # Get the database path (matching the path in connection.py)
    db_dir = Path(__file__).parent.parent.parent / 'data'
    db_path = db_dir / 'prepopulated.db'
    
    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        return False
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(models)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Add logo column if it doesn't exist
        if 'logo' not in columns:
            logger.info("Adding 'logo' column to models table...")
            cursor.execute("ALTER TABLE models ADD COLUMN logo TEXT")
            logger.info("Successfully added 'logo' column")
        else:
            logger.info("'logo' column already exists")
        
        # Add model_analysis column if it doesn't exist
        if 'model_analysis' not in columns:
            logger.info("Adding 'model_analysis' column to models table...")
            cursor.execute("ALTER TABLE models ADD COLUMN model_analysis TEXT")
            logger.info("Successfully added 'model_analysis' column")
        else:
            logger.info("'model_analysis' column already exists")
        
        # Commit the changes
        conn.commit()
        conn.close()
        
        logger.info("Migration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if conn:
            conn.close()
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the migration
    success = migrate_add_logo_and_analysis()
    
    if success:
        print("Migration completed successfully!")
    else:
        print("Migration failed. Check the logs for details.") 