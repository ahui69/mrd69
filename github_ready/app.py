#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOWY CZYSTY SERWER - Chat + Autonauka + Zegarek
Wszystko co potrzebne, nic więcej!
"""

import os, json, time, sqlite3, hashlib, re
from typing import List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ═══════════════════════════════════════════════════════
# KONFIGURACJA
# ═══════════════════════════════════════════════════════

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")

DB_PATH = "data/chat.db"
Path("data").mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════

app = FastAPI(title="MRD69 Simple")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════
# BAZA DANYCH (SQLite)
# ═══════════════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        content TEXT,
        ts REAL
    )""")
    conn.commit()
    conn.close()

init_db()

def db_add_message(role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO messages(role, content, ts) VALUES(?,?,?)",
                 (role, content, time.time()))
    conn.commit()
    conn.close()

def db_get_history(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT role, content, ts FROM messages ORDER BY ts DESC LIMIT ?",
                        (limit,)).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"], "ts": r["ts"]} for r in reversed(rows)]

# ═══════════════════════════════════════════════════════
# LLM CLIENT
# ═══════════════════════════════════════════════════════

def call_llm(messages: List[Dict], temperature: float = 0.7) -> str:
    """Wywołaj LLM"""
    if not LLM_API_KEY:
        return "(brak klucza LLM - ustaw LLM_API_KEY w .env)"
    
    import requests
    
    url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1500
    }
    
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"(błąd LLM: {str(e)})"

# ═══════════════════════════════════════════════════════
# AUTONAUKA - PROSTE WEB LEARNING
# ═══════════════════════════════════════════════════════

def web_search_simple(query: str) -> List[Dict]:
    """Proste wyszukiwanie w internecie (DuckDuckGo)"""
    try:
        from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
        return results
    except Exception as e:
        print(f"Web search error: {e}")
        return []

def web_learn_simple(query: str) -> Dict:
    """Ucz się z internetu i zwróć kontekst"""
    # 1. Szukaj w internecie
    results = web_search_simple(query)
    
    if not results:
        return {
            "query": query,
            "context": "(brak wyników wyszukiwania)",
            "sources": []
        }
    
    # 2. Zbierz snippety
    context_parts = []
    sources = []
    
    for r in results[:5]:
        context_parts.append(f"{r['title']}: {r['snippet']}")
        sources.append({"title": r['title'], "url": r['url']})
    
    context = "\n\n".join(context_parts)
    
    # 3. Poproś LLM o podsumowanie
    summary_prompt = [
        {"role": "system", "content": "Jesteś asystentem. Podsumuj informacje zwięźle po polsku."},
        {"role": "user", "content": f"Pytanie: {query}\n\nInformacje z internetu:\n{context}\n\nPodsumuj:"}
    ]
    
    summary = call_llm(summary_prompt, temperature=0.3)
    
    return {
        "query": query,
        "context": context,
        "summary": summary,
        "sources": sources
    }

# ═══════════════════════════════════════════════════════
# MODELE DANYCH
# ═══════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    use_web: bool = False

class WebLearnRequest(BaseModel):
    query: str

# ═══════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "time": int(time.time()),
        "llm_configured": bool(LLM_API_KEY)
    }

@app.get("/api/time")
def get_time():
    """Zwraca aktualny czas i datę"""
    import datetime
    now = datetime.datetime.now()
    return {
        "timestamp": int(time.time()),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "day_name": now.strftime("%A"),
        "month_name": now.strftime("%B")
    }

@app.post("/api/chat")
def chat(req: ChatRequest):
    """Główny endpoint czatu"""
    
    # 1. Zapisz wiadomość usera
    db_add_message("user", req.message)
    
    # 2. Pobierz historię
    history = db_get_history(limit=10)
    
    # 3. Przygotuj messages dla LLM
    messages = [
        {"role": "system", "content": "Jesteś pomocnym asystentem. Odpowiadasz po polsku, konkretnie i zwięźle."}
    ]
    
    # Dodaj historię
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    
    # 4. Jeśli user chce web search
    web_context = ""
    sources = []
    
    if req.use_web:
        web_data = web_learn_simple(req.message)
        web_context = web_data.get("summary", "")
        sources = web_data.get("sources", [])
        
        if web_context:
            messages.append({
                "role": "system",
                "content": f"Informacje z internetu:\n{web_context}"
            })
    
    # 5. Wywołaj LLM
    response = call_llm(messages, temperature=0.7)
    
    # 6. Zapisz odpowiedź
    db_add_message("assistant", response)
    
    return {
        "ok": True,
        "response": response,
        "sources": sources if sources else None
    }

@app.get("/api/history")
def history():
    """Zwróć historię czatu"""
    return {
        "ok": True,
        "messages": db_get_history(limit=50)
    }

@app.post("/api/learn")
def learn(req: WebLearnRequest):
    """Web learning - ucz się z internetu"""
    result = web_learn_simple(req.query)
    return {
        "ok": True,
        **result
    }

@app.delete("/api/history")
def clear_history():
    """Wyczyść historię"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages")
    conn.commit()
    conn.close()
    return {"ok": True, "message": "Historia wyczyszczona"}

# ═══════════════════════════════════════════════════════
# FRONTEND
# ═══════════════════════════════════════════════════════

FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MRD69 Chat - Asystent AI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: calc(100vh - 40px);
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            font-size: 24px;
            font-weight: 700;
        }
        
        .clock-date {
            text-align: right;
            font-size: 14px;
        }
        
        .clock {
            font-size: 28px;
            font-weight: 700;
            font-family: 'Courier New', monospace;
            letter-spacing: 2px;
        }
        
        .date {
            font-size: 12px;
            opacity: 0.9;
            margin-top: 4px;
        }
        
        .chat-box {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f5f5f5;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 18px;
            line-height: 1.5;
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .message.assistant .message-content {
            background: white;
            color: #333;
            border: 1px solid #ddd;
        }
        
        .message.system .message-content {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffc107;
            max-width: 85%;
            font-size: 13px;
        }
        
        .sources {
            margin-top: 10px;
            padding: 10px;
            background: #e8f4f8;
            border-radius: 8px;
            font-size: 12px;
        }
        
        .sources a {
            color: #667eea;
            text-decoration: none;
            display: block;
            margin: 4px 0;
        }
        
        .sources a:hover {
            text-decoration: underline;
        }
        
        .input-area {
            padding: 20px;
            background: white;
            border-top: 1px solid #ddd;
        }
        
        .input-controls {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .input-controls label {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 14px;
            cursor: pointer;
        }
        
        .input-controls input[type="checkbox"] {
            cursor: pointer;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
        }
        
        textarea {
            flex: 1;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 12px;
            font-size: 15px;
            font-family: inherit;
            resize: none;
            outline: none;
            transition: border 0.3s;
        }
        
        textarea:focus {
            border-color: #667eea;
        }
        
        button {
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .btn-clear {
            background: #dc3545;
            padding: 8px 16px;
            font-size: 13px;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #667eea;
        }
        
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 MRD69 Asystent</h1>
            <div class="clock-date">
                <div class="clock" id="clock">--:--:--</div>
                <div class="date" id="date">---- -- --</div>
            </div>
        </div>
        
        <div class="chat-box" id="chatBox">
            <div class="message system">
                <div class="message-content">
                    👋 Cześć! Jestem Twoim asystentem AI. Zadaj mi pytanie albo zaznacz "Szukaj w internecie" dla aktualnych informacji!
                </div>
            </div>
        </div>
        
        <div class="input-area">
            <div class="input-controls">
                <label>
                    <input type="checkbox" id="useWeb">
                    🌐 Szukaj w internecie
                </label>
                <button class="btn-clear" onclick="clearHistory()">🗑️ Wyczyść</button>
            </div>
            
            <div class="input-group">
                <textarea 
                    id="messageInput" 
                    rows="3" 
                    placeholder="Napisz wiadomość..."
                    onkeypress="if(event.key==='Enter' && !event.shiftKey){event.preventDefault();sendMessage();}"
                ></textarea>
                <button onclick="sendMessage()" id="sendBtn">Wyślij</button>
            </div>
        </div>
    </div>

    <script>
        const chatBox = document.getElementById('chatBox');
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        const useWebCheckbox = document.getElementById('useWeb');
        
        // ZEGAR I DATA
        function updateClock() {
            const now = new Date();
            
            // Czas
            const hours = String(now.getHours()).padStart(2, '0');
            const minutes = String(now.getMinutes()).padStart(2, '0');
            const seconds = String(now.getSeconds()).padStart(2, '0');
            document.getElementById('clock').textContent = `${hours}:${minutes}:${seconds}`;
            
            // Data
            const days = ['Niedziela', 'Poniedziałek', 'Wtorek', 'Środa', 'Czwartek', 'Piątek', 'Sobota'];
            const months = ['stycznia', 'lutego', 'marca', 'kwietnia', 'maja', 'czerwca',
                           'lipca', 'sierpnia', 'września', 'października', 'listopada', 'grudnia'];
            
            const dayName = days[now.getDay()];
            const day = now.getDate();
            const month = months[now.getMonth()];
            const year = now.getFullYear();
            
            document.getElementById('date').textContent = `${dayName}, ${day} ${month} ${year}`;
        }
        
        // Aktualizuj zegar co sekundę
        updateClock();
        setInterval(updateClock, 1000);
        
        // CHAT FUNKCJE
        function addMessage(role, content, sources = null) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${role}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            
            msgDiv.appendChild(contentDiv);
            
            // Dodaj źródła jeśli są
            if (sources && sources.length > 0) {
                const sourcesDiv = document.createElement('div');
                sourcesDiv.className = 'sources';
                sourcesDiv.innerHTML = '<strong>📚 Źródła:</strong><br>' + 
                    sources.map(s => `<a href="${s.url}" target="_blank">• ${s.title}</a>`).join('');
                msgDiv.appendChild(sourcesDiv);
            }
            
            chatBox.appendChild(msgDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            const useWeb = useWebCheckbox.checked;
            
            // Dodaj wiadomość usera
            addMessage('user', message);
            messageInput.value = '';
            
            // Pokaż loading
            sendBtn.disabled = true;
            sendBtn.innerHTML = '<div class="spinner"></div>';
            
            try {
                // Wyślij do API
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message: message,
                        use_web: useWeb
                    })
                });
                
                const data = await response.json();
                
                if (data.ok) {
                    addMessage('assistant', data.response, data.sources);
                } else {
                    addMessage('system', 'Błąd: ' + (data.error || 'Nieznany błąd'));
                }
            } catch (error) {
                addMessage('system', 'Błąd połączenia: ' + error.message);
            } finally {
                sendBtn.disabled = false;
                sendBtn.textContent = 'Wyślij';
                messageInput.focus();
            }
        }
        
        async function clearHistory() {
            if (!confirm('Wyczyścić całą historię czatu?')) return;
            
            try {
                await fetch('/api/history', {method: 'DELETE'});
                chatBox.innerHTML = '<div class="message system"><div class="message-content">Historia wyczyszczona! Zacznij nową rozmowę.</div></div>';
            } catch (error) {
                addMessage('system', 'Błąd czyszczenia: ' + error.message);
            }
        }
        
        // Załaduj historię przy starcie
        async function loadHistory() {
            try {
                const response = await fetch('/api/history');
                const data = await response.json();
                
                if (data.ok && data.messages.length > 0) {
                    for (const msg of data.messages) {
                        addMessage(msg.role, msg.content);
                    }
                }
            } catch (error) {
                console.error('Błąd ładowania historii:', error);
            }
        }
        
        loadHistory();
        messageInput.focus();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
def serve_frontend():
    return HTMLResponse(content=FRONTEND_HTML)

# ═══════════════════════════════════════════════════════
# START
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("🚀 Uruchamiam MRD69 Simple...")
    print("📖 Dokumentacja: http://localhost:8000/docs")
    print("🎨 Chat: http://localhost:8000/")
    uvicorn.run(app, host="0.0.0.0", port=8000)
