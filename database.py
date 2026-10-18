import sqlite3
import bcrypt
from datetime import datetime

DB_NAME = "users.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def create_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'patient',
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_name TEXT,
            image_path TEXT,
            prediction_date TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            action TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# ---------- Password helpers (bcrypt) ----------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ---------- Activity log ----------
def log_activity(username, action):
    conn = get_connection()
    conn.execute(
        "INSERT INTO activity_log (username, action, timestamp) VALUES (?, ?, ?)",
        (username, action, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()


# ---------- Register ----------
def register_user(username, name, email, phone, password):
    conn = get_connection()
    cursor = conn.cursor()
    pwd_hash = hash_password(password)
    try:
        cursor.execute("""
            INSERT INTO users (username, name, email, phone, password_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (username, name, email, phone, pwd_hash,
              datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError as e:
        err = str(e).lower()
        if "username" in err:
            return False, "Username already exists."
        if "email" in err:
            return False, "Email already registered."
        return False, "Unable to create account."
    finally:
        conn.close()


# ---------- Login ----------
def login_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, name, email, phone, password_hash FROM users WHERE username = ?",
        (username,),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    if verify_password(password, row[5]):
        return (row[0], row[1], row[2], row[3], row[4])
    return None


# ---------- Save prediction ----------
def save_prediction(username, disease, confidence, image_name, image_path=None):
    conn = get_connection()
    conn.execute("""
        INSERT INTO prediction_history
        (username, disease, confidence, image_name, image_path, prediction_date)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (username, disease, float(confidence), image_name, image_path,
          datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    log_activity(username, f"prediction:{disease}")


def get_prediction_history(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, disease, confidence, image_name, image_path, prediction_date
        FROM prediction_history WHERE username = ? ORDER BY id DESC
    """, (username,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def delete_prediction(prediction_id, username):
    conn = get_connection()
    conn.execute(
        "DELETE FROM prediction_history WHERE id = ? AND username = ?",
        (prediction_id, username),
    )
    conn.commit()
    conn.close()


def get_prediction_count(username=None):
    conn = get_connection()
    cursor = conn.cursor()
    if username:
        cursor.execute("SELECT COUNT(*) FROM prediction_history WHERE username = ?", (username,))
    else:
        cursor.execute("SELECT COUNT(*) FROM prediction_history")
    c = cursor.fetchone()[0]
    conn.close()
    return c


def get_user_profile(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT username, name, email, phone, created_at FROM users WHERE username = ?",
        (username,),
    )
    user = cursor.fetchone()
    conn.close()
    return user


def update_profile(current_username, new_username, new_name):
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE users SET username = ?, name = ? WHERE username = ?",
            (new_username, new_name, current_username),
        )
        conn.commit()
        return True, "Profile updated successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()


def change_password(username, old_password, new_password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    if not row or not verify_password(old_password, row[0]):
        conn.close()
        return False, "Current password is incorrect."
    new_hash = hash_password(new_password)
    conn.execute("UPDATE users SET password_hash = ? WHERE username = ?", (new_hash, username))
    conn.commit()
    conn.close()
    return True, "Password changed successfully."


def get_activity_log(username, limit=20):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT action, timestamp FROM activity_log
        WHERE username = ? ORDER BY id DESC LIMIT ?
    """, (username, limit))
    rows = cursor.fetchall()
    conn.close()
    return rows