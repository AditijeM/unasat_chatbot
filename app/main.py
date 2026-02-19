import os
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# DIT IS HET EXACTE PAD DAT DE NIEUWE ROUTER VERWACHT
ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"HTML Error: {e}"

@app.post("/chat")
async def chat(message: dict):
    user_input = message.get("text")
    
    # We gebruiken Llama 3.2 of Qwen, beide zijn 'native' op de nieuwe router
    payload = {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "messages": [
            {"role": "system", "content": "Je bent een UNASAT assistent."},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    print(f"\n--- VERBINDING MET DE NIEUWE ROUTER ---")
    try:
        # We praten nu tegen de router.huggingface.co
        response = requests.post(ROUTER_URL, headers=headers, json=payload, timeout=15)
        
        print(f"Router Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            reply = result['choices'][0]['message']['content']
            return {"reply": reply}
        else:
            # Als het misgaat, printen we de volledige reden van de router
            print(f"Router Error Body: {response.text}")
            return {"reply": f"Router Fout {response.status_code}: {response.text[:100]}"}
            
    except Exception as e:
        print(f"Netwerk Crash: {str(e)}")
        return {"reply": "Kon geen verbinding maken met de router."}