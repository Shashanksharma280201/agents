import sqlite3
import os
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "agents.db"
MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent reads
    return conn


def init_db() -> None:
    """Run all migration files in order. Safe to call on every startup."""
    conn = get_connection()
    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    with conn:
        for path in migration_files:
            conn.executescript(path.read_text())
    conn.close()
