import os
import pymysql
import pymysql.cursors
import bcrypt
from dotenv import load_dotenv

load_dotenv()

# Haal database configuratie uit environment variabelen (ingesteld in docker-compose)
DB_HOST = os.getenv("DB_HOST", "db")  # service name in docker-compose
DB_USER = os.getenv("DB_USER", "unasat_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "unasat_pass")
DB_NAME = os.getenv("DB_NAME", "unasat_db")

def get_db_connection():
    """Geeft een MySQL connectie terug met dictionaries als rows"""
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    return conn

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def init_db():
    """Maak tabellen aan als ze nog niet bestaan (wordt ook gedaan via init.sql)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Gebruiker tabel
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            full_name VARCHAR(100) NOT NULL
        )
    """)
    
    # Student resultaten tabel
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) NOT NULL,
            module VARCHAR(100) NOT NULL,
            grade DECIMAL(3,1),
            ec INT,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    """)
    
    # Rooster tabel
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schedule (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) NOT NULL,
            module VARCHAR(100) NOT NULL,
            day VARCHAR(20),
            time VARCHAR(20),
            room VARCHAR(20),
            FOREIGN KEY (username) REFERENCES users(username)
        )
    """)
    
    # Chat logs tabel (zonder foreign key naar users, want we willen logs bewaren zelfs als user verwijderd is)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_input TEXT,
            bot_reply TEXT,
            latency DECIMAL(5,2),
            feedback VARCHAR(10),
            username VARCHAR(50),
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Testdata toevoegen als student1 nog niet bestaat
    cursor.execute("SELECT * FROM users WHERE username = 'student1'")
    if not cursor.fetchone():
        hashed_pw = hash_password('unasat123')
        cursor.execute(
            "INSERT INTO users (username, password, full_name) VALUES (%s, %s, %s)",
            ('student1', hashed_pw, 'Jouw Naam')
        )
        cursor.execute(
            "INSERT INTO student_results (username, module, grade, ec) VALUES (%s, %s, %s, %s)",
            ('student1', 'Artificial Intelligence', 8.5, 6)
        )
        cursor.execute(
            "INSERT INTO student_results (username, module, grade, ec) VALUES (%s, %s, %s, %s)",
            ('student1', 'Software Engineering', 7.2, 4)
        )
        cursor.execute(
            "INSERT INTO schedule (username, module, day, time, room) VALUES (%s, %s, %s, %s, %s)",
            ('student1', 'Cloud Computing', 'Maandag', '18:00u', 'Lokaal 4.2')
        )
        print("--- TESTDATA TOEGEVOEGD (gehasht wachtwoord) ---")
    
    # Admin gebruiker toevoegen (als die nog niet bestaat)
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        hashed_pw = hash_password('admin123')
        cursor.execute(
            "INSERT INTO users (username, password, full_name) VALUES (%s, %s, %s)",
            ('admin', hashed_pw, 'Administrator')
        )
        print("--- ADMIN GEBRUIKER AANGEMAAKT (wachtwoord: admin123) ---")

    conn.commit()
    conn.close()



def verify_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    conn.close()
    if user and verify_password(password, user['password']):
        return user
    return None

def get_student_grades(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT module, grade FROM student_results WHERE username = %s", (username,))
    res = cursor.fetchall()
    conn.close()
    return res

def get_student_schedule(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT module, day, time, room FROM schedule WHERE username = %s", (username,))
    res = cursor.fetchall()   # ← fetchall i.p.v. fetchone
    conn.close()
    return res   # geeft een lijst (mogelijk leeg)

def log_chat(user_input, reply, latency, username):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_logs (user_input, bot_reply, latency, username) VALUES (%s, %s, %s, %s)",
        (user_input, reply, latency, username)
    )
    conn.commit()
    log_id = cursor.lastrowid
    conn.close()
    return log_id

def save_feedback(log_id, score):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE chat_logs SET feedback = %s WHERE id = %s", (score, log_id))
    conn.commit()
    conn.close()

def get_stats():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            COUNT(*) as total, 
            AVG(latency) as avg_lat,
            SUM(CASE WHEN feedback = 'good' THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN feedback = 'bad' THEN 1 ELSE 0 END) as negative 
        FROM chat_logs
    """)
    row = cursor.fetchone()
    conn.close()
    return row

def get_recent_logs(limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_logs ORDER BY id DESC LIMIT %s", (limit,))
    logs = cursor.fetchall()
    conn.close()
    return logs

def get_logs_by_user(username, limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM chat_logs WHERE username = %s ORDER BY id DESC LIMIT %s",
        (username, limit)
    )
    logs = cursor.fetchall()
    conn.close()
    return logs

def get_conversation_history(username, limit=5):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_input, bot_reply FROM chat_logs 
        WHERE username = %s 
        ORDER BY id DESC LIMIT %s
    """, (username, limit))
    rows = cursor.fetchall()
    conn.close()
    # Omgekeerde volgorde zodat oudste eerst komt
    return list(reversed(rows))  # nu van oud naar nieuw