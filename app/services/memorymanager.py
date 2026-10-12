import uuid
import sqlite3
import json
import os
from typing import Dict, List


DB_PATH = "memory_store/sessions.db"


class MemoryManager:
    """
    Persistent session-based conversation storage backed by SQLite.

    Fixes applied:
    1. session_exists() added — endpoints can now guard against fabricated session IDs.
    2. SQLite backend replaces the in-process dict so conversation history
       survives server restarts. FAISS vector indexes already persist to disk;
       this brings LLM history to the same durability level.
    3. Public API (create_session / add_message / get_history / clear_session)
       is unchanged — no call-site changes needed elsewhere.
    """

    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role      TEXT NOT NULL,
                    content   TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            conn.commit()

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id) VALUES (?)",
                (session_id,)
            )
            conn.commit()
        return session_id

    def session_exists(self, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
        return row is not None

    def add_message(self, session_id: str, role: str, content: str):
        # Auto-create session if called internally (e.g. from unit tests)
        # but endpoints.py now guards before reaching here.
        if not self.session_exists(session_id):
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
                    (session_id,)
                )
                conn.commit()

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content)
            )
            conn.commit()

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,)
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def clear_session(self, session_id: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()


# Global instance — safe because all state is DB-backed and session-keyed
memory_manager = MemoryManager()