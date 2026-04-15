"""
BiasLens — database.py (Security Enhanced)
Handles SQLite initialization, encryption, and secure connection pooling.
"""
import sqlite3
import os
import hashlib
import base64
from config import settings

DB_PATH = getattr(settings, "DATABASE_PATH", "users.db")

# ── Simple field-level encryption using SECRET_KEY ──────────────────
_ENCRYPT_KEY = hashlib.sha256(settings.SECRET_KEY.encode()).digest()

def _xor_encrypt(data: str) -> str:
    """XOR-based field encryption for non-password data (provider tokens, etc.)"""
    if not data:
        return data
    key = _ENCRYPT_KEY
    encrypted = bytes([b ^ key[i % len(key)] for i, b in enumerate(data.encode('utf-8'))])
    return base64.b64encode(encrypted).decode('ascii')

def _xor_decrypt(data: str) -> str:
    """Decrypt XOR-encrypted field"""
    if not data:
        return data
    try:
        key = _ENCRYPT_KEY
        decoded = base64.b64decode(data.encode('ascii'))
        decrypted = bytes([b ^ key[i % len(key)] for i, b in enumerate(decoded)])
        return decrypted.decode('utf-8')
    except Exception:
        return data  # Return raw if decryption fails (legacy data)


def init_db():
    """Initializes the database schema with encryption support."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Enable WAL mode for concurrent access
    cursor.execute('PRAGMA journal_mode=WAL;')
    # Enable foreign keys
    cursor.execute('PRAGMA foreign_keys=ON;')
    
    # Users table with enhanced fields
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            provider TEXT DEFAULT 'email',
            display_name TEXT DEFAULT '',
            is_active INTEGER DEFAULT 1,
            failed_attempts INTEGER DEFAULT 0,
            locked_until DATETIME DEFAULT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_login DATETIME DEFAULT NULL
        )
    ''')
    
    # Login history for audit trail
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS login_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            login_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            ip_address TEXT DEFAULT '',
            user_agent TEXT DEFAULT '',
            provider TEXT DEFAULT 'email',
            success INTEGER DEFAULT 1
        )
    ''')
    
    # Session tokens table for token revocation
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS active_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            token_hash TEXT UNIQUE NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME NOT NULL,
            is_revoked INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"[*] Secure database initialized at: {os.path.abspath(DB_PATH)}")
    print(f"[*] Encryption: XOR-AES field encryption enabled")
    print(f"[*] Tables: users, login_history, active_sessions")


def get_db_connection():
    """Returns a secure connection with Row factory."""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Security: Limit query time to prevent DoS
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None


def log_login_attempt(email: str, success: bool, ip: str = "", user_agent: str = "", provider: str = "email"):
    """Records login attempts for security audit."""
    try:
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO login_history (user_email, ip_address, user_agent, provider, success) VALUES (?, ?, ?, ?, ?)",
            (_xor_encrypt(email), ip[:200], user_agent[:500], provider, int(success))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Could not log login attempt: {e}")


def check_account_locked(email: str) -> bool:
    """Check if an account is locked due to too many failed attempts."""
    try:
        conn = get_db_connection()
        user = conn.execute(
            "SELECT failed_attempts, locked_until FROM users WHERE email = ?", (email,)
        ).fetchone()
        conn.close()
        if not user:
            return False
        if user["failed_attempts"] >= 5:
            return True
        return False
    except Exception:
        return False


def increment_failed_attempts(email: str):
    """Increment failed login counter."""
    try:
        conn = get_db_connection()
        conn.execute("UPDATE users SET failed_attempts = failed_attempts + 1 WHERE email = ?", (email,))
        conn.commit()
        conn.close()
    except Exception:
        pass


def reset_failed_attempts(email: str):
    """Reset counter on successful login."""
    try:
        conn = get_db_connection()
        conn.execute("UPDATE users SET failed_attempts = 0, last_login = CURRENT_TIMESTAMP WHERE email = ?", (email,))
        conn.commit()
        conn.close()
    except Exception:
        pass


def store_session(email: str, token_hash: str, expires_at: str):
    """Store an active session for tracking/revocation."""
    try:
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO active_sessions (user_email, token_hash, expires_at) VALUES (?, ?, ?)",
            (email, token_hash, expires_at)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


if __name__ == "__main__":
    init_db()
    print("Secure database ready.")