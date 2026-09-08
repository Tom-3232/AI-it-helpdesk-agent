import os
import sqlite3
import datetime
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "helpdesk.db"

def get_connection():
    """Returns a SQLite connection to helpdesk.db."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema if tables do not exist."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Tickets Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT UNIQUE NOT NULL,
                user_name TEXT NOT NULL,
                issue TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'OPEN',
                created_at TEXT NOT NULL
            )
        """)
        
        # User Memory Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                memory_value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, memory_key) ON CONFLICT REPLACE
            )
        """)
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB Error] Database initialization failed: {e}")

def insert_ticket(user_name: str, issue: str, priority: str, ticket_id: str = None) -> dict:
    """Inserts a new ticket record into SQLite."""
    init_db()
    if not ticket_id:
        import random
        num = random.randint(10000, 99999)
        ticket_id = f"TKT-2026-{num}"
        
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO tickets (ticket_id, user_name, issue, priority, status, created_at)
            VALUES (?, ?, ?, ?, 'OPEN', ?)
        """, (ticket_id, user_name, issue, priority.upper(), created_at))
        conn.commit()
        conn.close()
        return {
            "ticket_id": ticket_id,
            "user_name": user_name,
            "issue": issue,
            "priority": priority.upper(),
            "status": "OPEN",
            "created_at": created_at
        }
    except Exception as e:
        print(f"[DB Error] Failed to insert ticket: {e}")
        return {"error": str(e)}

def get_ticket(ticket_id: str) -> dict:
    """Retrieves a ticket by ticket_id."""
    init_db()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id.strip(),))
        row = cursor.fetchone()
        conn.close()
        if row:
            return dict(row)
        return None
    except Exception as e:
        print(f"[DB Error] Failed to get ticket: {e}")
        return None

def get_all_tickets(status: str = None) -> list:
    """Retrieves all tickets, optionally filtered by status ('OPEN' or 'CLOSED')."""
    init_db()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        if status:
            cursor.execute("SELECT * FROM tickets WHERE status = ? ORDER BY id DESC", (status.upper(),))
        else:
            cursor.execute("SELECT * FROM tickets ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[DB Error] Failed to get tickets: {e}")
        return []

def close_ticket(ticket_id: str) -> bool:
    """Updates status of a ticket to CLOSED."""
    init_db()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE tickets SET status = 'CLOSED' WHERE ticket_id = ?", (ticket_id.strip().upper(),))
        conn.commit()
        conn.close()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"[DB Error] Failed to close ticket: {e}")
        return False

def get_ticket_stats() -> dict:
    """Returns total open and closed ticket counts."""
    init_db()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT status, COUNT(*) as count FROM tickets GROUP BY status")
        rows = cursor.fetchall()
        conn.close()
        
        stats = {"OPEN": 0, "CLOSED": 0, "TOTAL": 0}
        for row in rows:
            st = row["status"].upper()
            stats[st] = row["count"]
            stats["TOTAL"] += row["count"]
        return stats
    except Exception as e:
        print(f"[DB Error] Failed to get ticket stats: {e}")
        return {"OPEN": 0, "CLOSED": 0, "TOTAL": 0}

def save_user_memory(user_id: str, memory_key: str, memory_value: str) -> bool:
    """Saves or updates a fact key-value pair for a user in long-term memory."""
    init_db()
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO user_memory (user_id, memory_key, memory_value, created_at)
            VALUES (?, ?, ?, ?)
        """, (user_id, memory_key.lower(), memory_value, created_at))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[DB Error] Failed to save memory: {e}")
        return False

def get_user_memory(user_id: str, memory_key: str = None) -> list:
    """Retrieves long-term memory facts for a user."""
    init_db()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        if memory_key:
            cursor.execute("SELECT * FROM user_memory WHERE user_id = ? AND memory_key = ?", (user_id, memory_key.lower()))
        else:
            cursor.execute("SELECT * FROM user_memory WHERE user_id = ?", (user_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[DB Error] Failed to get memory: {e}")
        return []
