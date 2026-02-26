import os
import time
import logging
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from pydantic import BaseModel

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

from models import ChatMessage, LoginData, FeedbackData
import database as db

# ------------------ CONFIG ------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unasat-ai-mentor")

app = FastAPI(title="UNASAT AI Student Mentor")
app.mount("/static", StaticFiles(directory="static"), name="static")
db.init_db()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# ------------------ STATE ------------------
active_conversations = {}

# ------------------ KENNISBANK ------------------
def load_knowledge_base():
    try:
        with open("unasat_kb.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "UNASAT (University of Applied Sciences and Technology) is gevestigd in Paramaribo, Suriname."

# ------------------ INTENT DETECTIE ------------------
def detect_intent(text: str):
    t = text.lower()
    # Alles wat met persoonlijke data te maken heeft (wordt nu behandeld door aparte endpoints, maar voor chat blijft)
    if any(w in t for w in ["cijfer", "resultaat", "punten", "rooster", "les", "planning", "gemiddelde", "schema"]):
        return "personal"
    # Studieadvies (blijft apart omdat het twee stappen heeft)
    if any(w in t for w in ["advies", "opleiding", "studie kiezen", "welke studie"]):
        return "advice_step_1"
    # Emotionele ondersteuning
    if any(w in t for w in ["stress", "overweldigd", "moeilijk", "twijfel", "motivatie"]):
        return "emotional_support"
    return "general"

# ------------------ SYSTEM PROMPT BUILDER ------------------
def build_system_prompt(mode: str, language: str, student_data=None):
    kb = load_knowledge_base()
    taal = language  # 'nl' of 'en'

    basis = f"""Je bent de officiële AI Student Mentor van de University of Applied Sciences and Technology (UNASAT) in Paramaribo, Suriname.

## COMMUNICATIESTIJL
- Professioneel maar toegankelijk.
- Antwoord in het {taal if taal == 'nl' else 'Engels'}.
- Gebruik altijd de volledige naam "University of Applied Sciences and Technology (UNASAT)" bij de eerste vermelding in een antwoord.
- Houd antwoorden bondig (5-8 zinnen), tenzij het om een uitgebreide analyse gaat.
- Als informatie niet in de knowledge base staat, zeg dan eerlijk: "Deze informatie is niet beschikbaar in de UNASAT-database." (of Engels equivalent).

## KENNISBANK
{['kb']}  # we voegen later de echte kb toe

"""

    # Specifieke instructies per mode
    if mode == "personal":
        extra = """
## PERSOONLIJKE STUDENTGEGEVENS
Je hebt toegang tot de volgende gegevens van de student:
- Cijfers (module en cijfer)
- Rooster (module, dag, tijd, lokaal)

## INSTRUCTIES
- Als de gebruiker vraagt naar zijn/haar cijfers, presenteer ze dan in een overzichtelijke **Markdown-tabel** met kolommen Module en Cijfer.
- Als de gebruiker vraagt naar het rooster, presenteer dat dan in een **Markdown-tabel** met kolommen Module, Dag, Tijd, Lokaal.
- Als de gebruiker vraagt om een gemiddelde of analyse, bereken dat dan stap voor stap en geef het resultaat. Gebruik eventueel een tabel voor overzicht.
- Als de gebruiker vraagt om een planning, maak dan een realistische weekplanning op basis van de vakken en eventuele zwakke punten.
- Gebruik de gegevens alleen als ze relevant zijn voor de vraag. Als data ontbreekt, zeg dat dan eerlijk.
"""
    elif mode == "advice_step_1":
        extra = """
## STUDIEADVIES – STAP 1
De gebruiker heeft interesse in studieadvies. Stel **2 à 3 gerichte vragen** om zijn/haar interesses, sterke punten en carrièredoelen te achterhalen. Geef **nog geen aanbeveling**.
"""
    elif mode == "advice_step_2":
        extra = """
## STUDIEADVIES – STAP 2
De gebruiker heeft de eerdere vragen beantwoord. Gebruik die antwoorden (terug te vinden in de gespreksgeschiedenis) en de kennisbank om **1 à 2 opleidingen** aan te bevelen. Leg uit waarom elke opleiding past bij de interesses en doelen van de gebruiker. Benoem ook praktische carrièrevoordelen.
"""
    elif mode == "emotional_support":
        extra = """
## EMOTIONELE ONDERSTEUNING
Toon begrip voor de situatie van de student. Normaliseer gevoelens van stress of twijfel. Geef kleine, haalbare stappen om weer grip te krijgen. Blijf realistisch en vermijd overdreven optimisme.
"""
    else:  # general
        extra = """
## ALGEMENE VRAGEN
Beantwoord vragen over UNASAT, de opleidingen, locatie, etc. uitsluitend op basis van de kennisbank. Gebruik duidelijke structuur (inleiding – kern – relevantie). Als de vraag niet beantwoord kan worden, verwijs dan naar de UNASAT-database.
"""

    # Combineer basis + extra + knowledge base
    full_prompt = basis + extra + f"\n## KENNISBANK\n{kb}"
    return full_prompt

# ------------------ MESSAGE BUILDER ------------------
def build_messages(system_prompt, history, user_input, student_data=None):
    messages = [{"role": "system", "content": system_prompt}]

    # Voeg student data toe als extra system message (optioneel)
    if student_data:
        data_str = ""
        if student_data.get("grades"):
            data_str += "CIJFERS:\n" + "\n".join([f"- {g['module']}: {g['grade']}" for g in student_data["grades"]]) + "\n"
        if student_data.get("schedule"):
            s = student_data["schedule"]
            data_str += f"VOLGENDE LES: {s['module']} op {s['day']} om {s['time']} in {s['room']}\n"
        messages.append({"role": "system", "content": f"STUDENT DATA:\n{data_str}"})

    # Voeg geschiedenis toe
    for item in history:
        messages.append({"role": "user", "content": item["user_input"]})
        messages.append({"role": "assistant", "content": item["bot_reply"]})

    # Huidige vraag
    messages.append({"role": "user", "content": user_input})
    return messages

# ------------------ GROQ CALL (8B) ------------------
def call_groq(messages):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "max_tokens": 800,
        "temperature": 0.2,
        "top_p": 0.9
    }
    response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=15)
    return response.json()["choices"][0]["message"]["content"]

# ------------------ CHAT ROUTE ------------------
@app.post("/chat")
@limiter.limit("10/minute")
async def chat(message: ChatMessage, request: Request):
    start = time.time()
    user_input = message.text
    username = message.username

    # Taaldetectie
    try:
        language = detect(user_input)
    except:
        language = "nl"

    # Geschiedenis (max 6 berichten voor context)
    history = db.get_conversation_history(username, limit=6)

    # Intentie bepalen
    intent = detect_intent(user_input)
    mode = intent

    # Data ophalen voor persoonlijke vragen (indien nodig)
    student_data = None
    if intent == "personal":
        grades = db.get_student_grades(username)
        schedule = db.get_student_schedule(username)
        student_data = {"grades": grades, "schedule": schedule}

    # Adviesflow (twee stappen)
    if username in active_conversations and active_conversations[username] == "waiting":
        mode = "advice_step_2"
        del active_conversations[username]
    elif intent == "advice_step_1":
        active_conversations[username] = "waiting"

    # Bouw system prompt en messages
    system_prompt = build_system_prompt(mode, language, student_data)
    messages = build_messages(system_prompt, history, user_input, student_data)

    try:
        reply = call_groq(messages)
    except Exception as e:
        logger.error(f"Groq call failed: {e}")
        reply = "Er is een technische storing bij de University of Applied Sciences and Technology (UNASAT). Probeer het later opnieuw."

    latency = round(time.time() - start, 2)
    log_id = db.log_chat(user_input, reply, latency, username)
    return {"reply": reply, "latency": latency, "id": log_id}

# ------------------ NIEUWE DATA-ONTSLUITING ENDPOINTS ------------------
@app.get("/grades")
async def get_grades(username: str):
    """Haal alle cijfers van een student op (zonder AI)"""
    grades = db.get_student_grades(username)
    if not grades:
        return []
    return grades

@app.get("/schedule")
async def get_schedule(username: str):
    """Haal het rooster van een student op (zonder AI)"""
    schedule = db.get_student_schedule(username)
    if not schedule:
        return None
    return schedule

# ------------------ OVERIGE ROUTES (ONVERANDERD) ------------------
@app.post("/feedback")
async def feedback(data: FeedbackData):
    db.save_feedback(data.id, data.score)
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/login")
async def login(data: LoginData):
    user = db.verify_user(data.username, data.password)
    if user:
        return {"full_name": user['full_name'], "username": user['username']}
    raise HTTPException(status_code=401, detail="Fout")

@app.get("/stats")
async def stats():
    row = db.get_stats()
    return {
        "total": row['total'] or 0,
        "latency": round(row['avg_lat'] or 0, 2),
        "good": row['positive'] or 0,
        "bad": row['negative'] or 0
    }

@app.get("/logs")
async def get_logs():
    logs = db.get_recent_logs()
    result = []
    for row in logs:
        d = dict(row)
        result.append({
            "id": d.get('id'),
            "user_input": d.get('user_input'),
            "bot_reply": d.get('bot_reply'),
            "latency": d.get('latency'),
            "feedback": d.get('feedback'),
            "username": d.get('username'),
            "timestamp": d.get('timestamp')
        })
    return result

@app.get("/my-logs")
async def my_logs(username: str):
    try:
        logs = db.get_logs_by_user(username)
        result = []
        for row in logs:
            d = dict(row)
            result.append({
                "id": d.get('id'),
                "user_input": d.get('user_input'),
                "bot_reply": d.get('bot_reply'),
                "latency": d.get('latency'),
                "feedback": d.get('feedback'),
                "timestamp": d.get('timestamp')
            })
        return result
    except Exception as e:
        logger.error(f"Fout in /my-logs voor {username}: {e}")
        raise HTTPException(status_code=500, detail=f"Databasefout: {str(e)}")

@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    with open("static/admin.html", "r", encoding="utf-8") as f:
        return f.read()

class NewUserData(BaseModel):
    username: str
    password: str
    full_name: str
    grades: list[dict]
    schedule: list[dict]

@app.post("/admin/create-user")
async def create_user(data: NewUserData):
    try:
        hashed_pw = db.hash_password(data.password)
        conn = db.get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password, full_name) VALUES (%s, %s, %s)",
            (data.username, hashed_pw, data.full_name)
        )
        for g in data.grades:
            cursor.execute(
                "INSERT INTO student_results (username, module, grade, ec) VALUES (%s, %s, %s, %s)",
                (data.username, g['module'], g['grade'], g['ec'])
            )
        for s in data.schedule:
            cursor.execute(
                "INSERT INTO schedule (username, module, day, time, room) VALUES (%s, %s, %s, %s, %s)",
                (data.username, s['module'], s['day'], s['time'], s['room'])
            )
        conn.commit()
        conn.close()
        return {"status": "ok", "message": f"Gebruiker {data.username} aangemaakt"}
    except Exception as e:
        raise HTTPException(statuscode=500, detail=str(e))