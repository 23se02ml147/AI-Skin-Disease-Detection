import psycopg2  # MUST be imported first to prevent TensorFlow SSL crash
import os
import bcrypt
from psycopg2 import extras
from datetime import datetime

# ============================================================
# DATABASE CONNECTION SETUP
# ============================================================

# Railway provides the DATABASE_URL environment variable automatically
DATABASE_URL = os.getenv("DATABASE_URL")

# Fix for Railway's Postgres URL scheme (SQLAlchemy/psycopg2 needs 'postgresql://')
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

def get_connection():
    # Connect to PostgreSQL using the Railway URL
    return psycopg2.connect(DATABASE_URL)


def create_database():
    conn = get_connection()
    cursor = conn.cursor()

    # PostgreSQL uses SERIAL instead of AUTOINCREMENT
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
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
            id SERIAL PRIMARY KEY,
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
            id SERIAL PRIMARY KEY,
            username TEXT NOT NULL,
            action TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()


# ============================================================
# PASSWORD HELPERS (bcrypt)
# ============================================================
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ============================================================
# ACTIVITY LOG
# ============================================================
def log_activity(username, action):
    conn = get_connection()
    cursor = conn.cursor()
    # Changed ? to %s for Postgres
    cursor.execute(
        "INSERT INTO activity_log (username, action, timestamp) VALUES (%s, %s, %s)",
        (username, action, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    cursor.close()
    conn.close()


# ============================================================
# REGISTER
# ============================================================
def register_user(username, name, email, phone, password):
    conn = get_connection()
    cursor = conn.cursor()
    pwd_hash = hash_password(password)
    try:
        cursor.execute("""
            INSERT INTO users (username, name, email, phone, password_hash, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (username, name, email, phone, pwd_hash,
              datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        return True, "Account created successfully."
    except psycopg2.IntegrityError as e:
        conn.rollback() # Important in Postgres to clear the error state
        err = str(e).lower()
        if "username" in err:
            return False, "Username already exists."
        if "email" in err:
            return False, "Email already registered."
        return False, "Unable to create account."
    finally:
        cursor.close()
        conn.close()


# ============================================================
# LOGIN
# ============================================================
def login_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, name, email, phone, password_hash FROM users WHERE username = %s",
        (username,),
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if not row:
        return None
    if verify_password(password, row[5]):
        return (row[0], row[1], row[2], row[3], row[4])
    return None


# ============================================================
# SAVE PREDICTION
# ============================================================
def save_prediction(username, disease, confidence, image_name, image_path=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO prediction_history
        (username, disease, confidence, image_name, image_path, prediction_date)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (username, disease, float(confidence), image_name, image_path,
          datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    cursor.close()
    conn.close()
    log_activity(username, f"prediction:{disease}")


def get_prediction_history(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, disease, confidence, image_name, image_path, prediction_date
        FROM prediction_history WHERE username = %s ORDER BY id DESC
    """, (username,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def delete_prediction(prediction_id, username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM prediction_history WHERE id = %s AND username = %s",
        (prediction_id, username),
    )
    conn.commit()
    cursor.close()
    conn.close()


def get_prediction_count(username=None):
    conn = get_connection()
    cursor = conn.cursor()
    if username:
        cursor.execute("SELECT COUNT(*) FROM prediction_history WHERE username = %s", (username,))
    else:
        cursor.execute("SELECT COUNT(*) FROM prediction_history")
    c = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return c


# ============================================================
# USER PROFILE
# ============================================================
def get_user_profile(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT username, name, email, phone, created_at FROM users WHERE username = %s",
        (username,),
    )
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user


def update_profile(current_username, new_username, new_name):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE users SET username = %s, name = %s WHERE username = %s",
            (new_username, new_name, current_username),
        )
        conn.commit()
        return True, "Profile updated successfully."
    except psycopg2.IntegrityError:
        conn.rollback()
        return False, "Username already exists."
    finally:
        cursor.close()
        conn.close()


# ============================================================
# CHANGE PASSWORD
# ============================================================
def change_password(username, old_password, new_password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username = %s", (username,))
    row = cursor.fetchone()
    if not row or not verify_password(old_password, row[0]):
        cursor.close()
        conn.close()
        return False, "Current password is incorrect."
        
    new_hash = hash_password(new_password)
    cursor.execute("UPDATE users SET password_hash = %s WHERE username = %s", (new_hash, username))
    conn.commit()
    cursor.close()
    conn.close()
    return True, "Password changed successfully."


# ============================================================
# ACTIVITY LOG RETRIEVAL
# ============================================================
def get_activity_log(username, limit=20):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT action, timestamp FROM activity_log
        WHERE username = %s ORDER BY id DESC LIMIT %s
    """, (username, limit))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows