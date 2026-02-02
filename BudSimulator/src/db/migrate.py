"""
Versioned database migration framework for BudSimulator.

Each migration is a function registered with a version number. Migrations run
in order and are tracked in a ``migration_history`` table so they only execute
once.

Usage:
    from src.db.migrate import run_all_migrations
    run_all_migrations()  # safe to call on every startup
"""

import sqlite3
import logging
from datetime import datetime
from typing import Callable, Dict, List, Tuple

from .connection import DatabaseConnection

logger = logging.getLogger(__name__)

# Registry: version -> (description, migration_fn)
_MIGRATIONS: Dict[int, Tuple[str, Callable[[sqlite3.Cursor], None]]] = {}


def migration(version: int, description: str):
    """Decorator to register a migration function.

    Args:
        version: Positive integer. Must be unique across all migrations.
        description: Human-readable description of the migration.
    """
    def decorator(fn: Callable[[sqlite3.Cursor], None]):
        if version in _MIGRATIONS:
            raise ValueError(f"Duplicate migration version {version}")
        _MIGRATIONS[version] = (description, fn)
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Migration definitions — add new migrations below in ascending version order.
# ---------------------------------------------------------------------------

@migration(1, "Add logo and model_analysis columns to models table")
def _m001_add_logo_and_analysis(cursor: sqlite3.Cursor) -> None:
    cursor.execute("PRAGMA table_info(models)")
    columns = {col[1] for col in cursor.fetchall()}

    if "logo" not in columns:
        cursor.execute("ALTER TABLE models ADD COLUMN logo TEXT")
    if "model_analysis" not in columns:
        cursor.execute("ALTER TABLE models ADD COLUMN model_analysis TEXT")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _ensure_migration_table(cursor: sqlite3.Cursor) -> None:
    """Create the migration history table if it doesn't exist."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS migration_history (
            version INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
    """)


def _applied_versions(cursor: sqlite3.Cursor) -> set:
    """Return the set of already-applied migration versions."""
    cursor.execute("SELECT version FROM migration_history")
    return {row[0] for row in cursor.fetchall()}


def run_all_migrations() -> int:
    """Run all pending migrations in version order.

    Returns:
        Number of migrations applied.
    """
    db = DatabaseConnection()
    applied = 0

    with db.get_connection() as conn:
        cursor = conn.cursor()
        _ensure_migration_table(cursor)
        already_done = _applied_versions(cursor)

        for version in sorted(_MIGRATIONS.keys()):
            if version in already_done:
                continue

            description, fn = _MIGRATIONS[version]
            logger.info("Applying migration v%d: %s", version, description)
            try:
                fn(cursor)
                cursor.execute(
                    "INSERT INTO migration_history (version, description, applied_at) VALUES (?, ?, ?)",
                    (version, description, datetime.utcnow().isoformat()),
                )
                conn.commit()
                applied += 1
                logger.info("Migration v%d applied successfully", version)
            except Exception:
                conn.rollback()
                logger.exception("Migration v%d failed, rolling back", version)
                raise

    if applied == 0:
        logger.info("All migrations already applied")
    else:
        logger.info("Applied %d migration(s)", applied)

    return applied


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = run_all_migrations()
    print(f"Applied {count} migration(s).")
