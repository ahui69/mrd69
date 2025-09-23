#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/mrd69"
APP="$ROOT/webapp"
STATIC="$APP/static"
UPLOADS="$ROOT/uploads"
HIST="$ROOT/history"
SEED="${KNOWLEDGE_SEED:-$ROOT/data/sq3/seed.jsonl}"

mkdir -p "$APP" "$STATIC" "$UPLOADS" "$HIST"

# ====== server.py (FastAPI) ======
cat <<'PY' > "$ROOT/server.py"
from __future__ import annotations
import os, json, time, uuid, re, shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- katalogi ---
ROOT = Path("/workspace/mrd69")
APP = ROOT / "webapp"
STATIC = APP / "static"
UPLOADS = ROOT / "uploads"
HIST = ROOT / "history"
SEED_PATH = Path(os.getenv("KNOWLEDGE_SEED", str(ROOT / "data" / "sq3" / "seed.jsonl")))

for p in (UPLOADS, HIST, STATIC):
    p.mkdir(parents=True, exist_ok=True)

# --- opcjonalne moduły projektu ---
def _load_llm():
    try:
        from llm_client import chat as llm_chat  # prefer twoje API
        return llm_chat
    except Exception:
        # twardy fallback na DeepInfra/OpenAI-compatible z env
        import requests
        BASE = (os.getenv("LLM_BASE_URL") or "https://api.deepinfra.com/v1/openai").rstrip("/")
        KEY = (os.getenv("LLM_API_KEY") or "").strip()
        MODEL = (os.getenv("LLM_MODEL") or "zai-org/GLM-4.5-Air").strip()
        TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
        def _chat(messages, temperature=0.5, max_tokens=900):
            if not KEY:
                # trywialny echo w skrajności
                return "\n".join([m["content"] for m in messages if m["role"]=="user"])
            url = f"{BASE}/chat/completions"
            r = requests.post(url, headers={
                "Authorization": f"Bearer {KEY}",
                "Content-Type": "application/json"
            }, json={
                "model": MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }, timeout=TIMEOUT)
            r.raise_for_status()
            j = r.json()
            return ((j.get("choices") or [{}])[0].get("message") or {}).get("content","")
        return _chat
LLM = _load_llm()

try:
    from writing_all_pro import vinted_listing as vinted_gen  # twoje PRO
except Exception:
    def vinted_gen(**kw):
        return {"title_A":"(brak writing_all_pro)","title_B":"","price":0,"price_range":(0,0),
                "description_md":"Moduł writing_all_pro nie dostępny.","hashtags":[]}

# ===== Wiedza startowa =====
KnowledgeItem = Dict[str, Any]
KNOWLEDGE: List[KnowledgeItem] = []
def load_seed() -> List[KnowledgeItem]:
    global KNOWLEDGE
    KNOWLEDGE = []
    if not SEED_PATH.exists():
        return KNOWLEDGE
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: 
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                obj = {"text": ln}
            if "text" not in obj:
                # scal różne pola w jeden tekst
                obj["text"] = " ".join(str(v) for v in obj.values() if isinstance(v,(str,int,float)))
            KNOWLEDGE.append(obj)
    return KNOWLEDGE
load_seed()

def _clean_tokens(s:str)->List[str]:
    return re.findall(r"\w+", (s or "").lower())

def knn(query:str, k:int=6)->List[str]:
    """Lekki RAG bez bibliotek: punktuje po overlapie tokenów."""
    q = set(_clean_tokens(query))
    scored=[]
    for it in KNOWLEDGE[:5000]:
        txt = str(it.get("text",""))
        tset = set(_clean_tokens(txt))
        score = len(q & tset)
        if score>0:
            scored.append((score, txt[:2000]))
    scored.sort(key=lambda x:-x[0])
    return [t for _,t in scored[:k]]

# ===== Historia =====
def hist_path(cid:str)->Path:
    return HIST / f"{cid}.jsonl"

def new_conversation()->str:
    cid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    hist_path(cid).write_text("", encoding="utf-8")
    return cid

def list_conversations()->List[Dict[str,Any]]:
    out=[]
    for p in sorted(HIST.glob("*.jsonl"), reverse=True):
        try:
            size = p.stat().st_size
            out.append({"id": p.stem, "size": size, "mtime": int(p.stat().st_mtime)})
        except Exception:
            pass
    return out

def append_msg(cid:str, role:str, content:str, meta:Optional[dict]=None):
    line = {"ts": int(time.time()), "role": role, "content": content}
    if meta: line["meta"]=meta
    with open(hist_path(cid), "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False)+"\n")

def load_history(cid:str)->List[Dict[str,Any]]:
    p = hist_path(cid)
    if not p.exists(): return []
    out=[]
    with open(p,"r",encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out

# ===== FastAPI =====
app = FastAPI(title="Mordzix Assistant", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("ALLOWED_ORIGINS","*")],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# statyki (frontend)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

@app.get("/", response_class=HTMLResponse)
def root_index():
    index = (STATIC/"index.html")
    if index.exists(): 
        return index.read_text(encoding="utf-8")
    return "<h1>Mordzix Assistant</h1>"

# --- Konwersacje
@app.post("/api/new")
def api_new():
    cid = new_conversation()
    return {"id": cid, "knowledge_loaded": len(KNOWLEDGE)}

@app.get("/api/threads")
def api_threads():
    return {"items": list_conversations()}

@app.get("/api/threads/{cid}")
def api_thread(cid:str):
    return {"id": cid, "messages": load_history(cid)}

# --- Uploady (nie wysyłamy automatycznie)
@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...), cid: str = Form(...)):
    fname = f"{int(time.time())}_{re.sub(r'[^a-zA-Z0-9._-]','_',file.filename)}"
    dest = UPLOADS / fname
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    append_msg(cid, "system", f"[UPLOAD] {fname}", {"file": fname})
    return {"ok": True, "file": fname, "url": f"/files/{fname}"}

@app.get("/files/{fname}")
def api_file(fname:str):
    p = UPLOADS / fname
    if not p.exists():
        return JSONResponse({"error":"not found"}, status_code=404)
    return FileResponse(str(p))

# --- RAG + czat
@app.post("/api/chat")
def api_chat(payload: Dict[str,Any] = Body(...)):
    cid = payload.get("cid") or new_conversation()
    msg = (payload.get("message") or "").strip()
    if not msg:
        return {"cid": cid, "reply": ""}

    # kontekst z wiedzy
    rag_bits = knn(msg, k=6)
    sys_prompt = (
        "Jesteś pomocnym asystentem Mordzix PRO ULTRA. "
        "Używaj wiedzy dostarczonej w <WIEDZA/> tylko jeśli pasuje. "
        "Odpowiadaj po polsku, krótko i konkretnie.\n\n"
        "<WIEDZA>\n" + "\n---\n".join(rag_bits) + "\n</WIEDZA>"
    )
    history_msgs = [{"role":"system","content": sys_prompt}]
    # opcjonalnie możemy dodać skrót historii
    hist = load_history(cid)[-8:]
    for h in hist:
        if h.get("role") in ("user","assistant"):
            history_msgs.append({"role": h["role"], "content": h["content"]})

    history_msgs.append({"role":"user","content": msg})
    try:
        reply = LLM(history_msgs, temperature=float(os.getenv("CHAT_TEMP","0.6")), max_tokens=1100)
    except Exception as e:
        reply = f"(Błąd LLM: {e})"

    append_msg(cid, "user", msg)
    append_msg(cid, "assistant", reply)
    return {"cid": cid, "reply": reply}

# --- Vinted
@app.post("/api/vinted")
def api_vinted(payload: Dict[str,Any] = Body(...)):
    res = vinted_gen(
        brand=payload.get("brand",""),
        item=payload.get("item",""),
        cond=payload.get("cond",""),
        size=payload.get("size",""),
        color=payload.get("color",""),
        material=payload.get("material",""),
        model=payload.get("model",""),
        tier_hint=payload.get("tier"),
        base_price=payload.get("base"),
    )
    return res

# --- Travel (link do trasy)
@app.get("/api/travel/route")
def api_route(orig:str, dest:str, mode:str="driving"):
    q = f"https://www.google.com/maps/dir/?api=1&origin={orig}&destination={dest}&travelmode={mode}"
    return {"url": q}

# --- Crypto (prosty Coingecko public)
@app.get("/api/crypto/price")
def api_crypto_price(symbol:str="btc", vs:str="usd"):
    import requests
    sym = symbol.lower()
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price",
                         params={"ids": sym, "vs_currencies": vs}, timeout=20)
        data = r.json()
        return {"symbol": sym, "vs": vs, "price": data.get(sym,{}).get(vs)}
    except Exception as e:
        return {"symbol": sym, "vs": vs, "error": str(e)}

# --- Seed reload
@app.post("/api/reload-seed")
def api_reload_seed():
    n = len(load_seed())
    return {"reloaded": n}

# --- Zdrowie
@app.get("/api/health")
def api_health():
    return {"ok": True, "knowledge": len(KNOWLEDGE)}
PY

# ====== Frontend (index.html) ======
cat <<'HTML' > "$STATIC/index.html"
<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no"/>
<title>Mordzix Assistant</title>
<link rel="stylesheet" href="/static/style.css"/>
</head>
<body>
<div id="app">
  <aside id="sidebar">
    <header>
      <h2>Rozmowy</h2>
      <button id="newChat">+ Nowa</button>
    </header>
    <ul id="threads"></ul>
    <footer>
      <button id="reloadSeed" title="Przeładuj wiedzę">🔄 seed</button>
      <a href="/api/health" target="_blank">health</a>
    </footer>
  </aside>

  <main id="chat">
    <header id="topbar">
      <div class="title">Mordzix PRO ULTRA</div>
      <div id="mood">🙂</div>
    </header>

    <section id="messages"></section>

    <div id="composer">
      <label class="addfile">
        📎
        <input type="file" id="fileInput" multiple />
      </label>
      <input id="msg" type="text" placeholder="Napisz wiadomość..." autocomplete="off"/>
      <button id="mic" title="Mów">🎤</button>
      <button id="send">Wyślij</button>
    </div>

    <div id="attachPreview"></div>
    <div id="typing" class="hidden"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>
  </main>
</div>
<script src="/static/app.js"></script>
</body>
</html>
HTML

# ====== Frontend (style.css) ======
cat <<'CSS' > "$STATIC/style.css"
:root{
  --bg:#0f1115; --panel:#151923; --muted:#aab; --text:#e9edf1; --accent:#6ee7b7;
  --me:#2b3949; --bot:#1c2433;
}
*{box-sizing:border-box}
html,body{height:100%;margin:0;background:var(--bg);color:var(--text);font:16px/1.45 -apple-system,system-ui,Segoe UI,Roboto,Ubuntu}
#app{display:flex; height:100vh; width:100vw; overflow:hidden}

#sidebar{width:300px;background:var(--panel);border-right:1px solid #1f2734;display:flex;flex-direction:column}
#sidebar header{display:flex;align-items:center;justify-content:space-between;padding:12px 14px;border-bottom:1px solid #1f2734}
#sidebar h2{margin:0;font-size:18px}
#sidebar button{background:#243044;color:#fff;border:0;border-radius:6px;padding:6px 10px;cursor:pointer}
#sidebar ul{list-style:none;margin:0;padding:0;overflow:auto}
#sidebar li{padding:10px 12px;border-bottom:1px solid #1f2734;cursor:pointer}
#sidebar li:hover{background:#1a2230}
#sidebar footer{padding:10px 12px;border-top:1px solid #1f2734;display:flex;gap:10px}

#chat{flex:1;display:flex;flex-direction:column}
#topbar{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border-bottom:1px solid #1f2734;background:#0f1420}
#messages{flex:1; overflow:auto; padding:18px 18px 10px}

.msg{max-width:70%;padding:10px 12px;border-radius:12px;margin:6px 0;white-space:pre-wrap}
.msg.bot{background:var(--bot);text-align:left}
.msg.me{background:var(--me);margin-left:auto;text-align:right}
#typing{display:flex;gap:5px; padding:10px; margin:4px 0 12px; width:80px}
.hidden{display:none}
.dot{width:8px;height:8px;background:#9ac;display:inline-block;border-radius:50%;animation:blink 1.2s infinite}
.dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,80%,100%{opacity:.2}40%{opacity:1}}

#composer{display:flex;gap:8px;padding:10px;border-top:1px solid #1f2734;background:#0f1420}
#composer input[type=text]{flex:1;border:1px solid #243044;background:#101826;color:#fff;border-radius:8px;padding:12px 10px;font-size:16px}
#composer button{border:0;background:#2a3850;color:#fff;padding:10px 12px;border-radius:8px}
#composer .addfile{display:flex;align-items:center;justify-content:center;width:42px;height:42px;border:1px dashed #2a3850;border-radius:8px;cursor:pointer}
#composer input[type=file]{display:none}

#attachPreview{display:flex;flex-wrap:wrap;gap:8px;padding:0 12px 8px}
.badge{background:#223149;border:1px solid #2a3b5a;color:#cfe;border-radius:6px;padding:4px 8px;font-size:12px}

@media (max-width:820px){
  #sidebar{width:0;position:absolute;z-index:5;left:0;top:0;bottom:0;transition:width .2s}
}
CSS

# ====== Frontend (app.js) ======
cat <<'JS' > "$STATIC/app.js"
let state = {
  cid: null,
  attachments: [],
  speaking: false,
};

const $ = (q)=>document.querySelector(q);
const $$ = (q)=>document.querySelectorAll(q);

async function api(path, opts={}){
  const r = await fetch(path, {headers:{'Content-Type':'application/json'}, ...opts});
  if(!r.ok) throw new Error(await r.text());
  return await r.json();
}

function addMsg(role, text){
  const div = document.createElement('div');
  div.className = `msg ${role==='user'?'me':'bot'}`;
  div.textContent = text;
  $("#messages").appendChild(div);
  $("#messages").scrollTop = $("#messages").scrollHeight;
}

async function loadThreads(){
  const data = await api('/api/threads');
  const ul = $("#threads"); ul.innerHTML="";
  for(const t of data.items){
    const li = document.createElement('li');
    li.textContent = `${t.id} (${Math.round(t.size/1024)}kB)`;
    li.onclick = async ()=>{
      state.cid = t.id;
      $("#messages").innerHTML = "";
      const thr = await api(`/api/threads/${t.id}`);
      for(const m of thr.messages){
        if(m.role==='user' || m.role==='assistant') addMsg(m.role, m.content);
      }
    };
    ul.appendChild(li);
  }
}

async function ensureChat(){
  if(state.cid) return state.cid;
  const d = await api('/api/new', {method:'POST', body:'{}'});
  state.cid = d.id;
  return state.cid;
}

async function send(){
  const msg = $("#msg").value.trim();
  if(!msg && state.attachments.length===0) return;
  const cid = await ensureChat();

  // upload w tle (jeśli są)
  if(state.attachments.length){
    for(const f of state.attachments){
      const fd = new FormData();
      fd.append('file', f);
      fd.append('cid', cid);
      await fetch('/api/upload', {method:'POST', body:fd});
    }
    state.attachments = [];
    $("#attachPreview").innerHTML="";
  }

  addMsg('user', msg);
  $("#msg").value="";

  $("#typing").classList.remove('hidden');
  try{
    const res = await api('/api/chat', {method:'POST', body: JSON.stringify({cid, message: msg})});
    $("#typing").classList.add('hidden');
    addMsg('assistant', res.reply || "(pusto)");
  }catch(e){
    $("#typing").classList.add('hidden');
    addMsg('assistant', "❌ " + e.message);
  }
}

$("#send").onclick = send;
$("#msg").addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ e.preventDefault(); send(); } });

$("#newChat").onclick = async ()=>{ state.cid = null; $("#messages").innerHTML=""; await ensureChat(); await loadThreads(); };
$("#reloadSeed").onclick = async ()=>{ await api('/api/reload-seed',{method:'POST', body:'{}'}); addMsg('assistant','🔄 Załadowano wiedzę seed.jsonl'); };

$("#fileInput").onchange = (e)=>{
  const files = Array.from(e.target.files||[]);
  state.attachments.push(...files);
  const wrap = $("#attachPreview");
  for(const f of files){
    const b = document.createElement('span');
    b.className='badge'; b.textContent = f.name;
    wrap.appendChild(b);
  }
};

(function micInit(){
  const btn = $("#mic");
  const has = 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
  if(!has){ btn.disabled=true; btn.title="Brak Speech API w tej przeglądarce"; return; }
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const rec = new SR();
  rec.lang = 'pl-PL'; rec.interimResults = true; rec.continuous = true;

  btn.onclick = ()=>{
    if(!state.speaking){ rec.start(); state.speaking=true; btn.textContent="⏹"; }
    else{ rec.stop(); state.speaking=false; btn.textContent="🎤"; }
  };
  rec.onresult = (ev)=>{
    let final = ""; let interim="";
    for(const r of ev.results){
      const t = r[0].transcript;
      if(r.isFinal) final += t + " ";
      else interim += t;
    }
    $("#msg").value = (final || interim).trim();
  };
  rec.onend = ()=>{ state.speaking=false; $("#mic").textContent="🎤"; };
})();

(async function boot(){
  // nowa rozmowa + wczytane knowledge
  await ensureChat();
  await loadThreads();
  // ping zdrowia
  try{
    const h = await api('/api/health');
    addMsg('assistant', `Witaj 👋 (wiedza: ${h.knowledge})`);
  }catch{}
})();
JS

# ====== Procfile startowy ======
cat <<'SH' > "$ROOT/start.sh"
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/mrd69
# wymagane paczki
python3 -m pip install --no-cache-dir --upgrade pip >/dev/null
python3 -m pip install --no-cache-dir fastapi uvicorn[standard] python-multipart requests python-dotenv >/dev/null
# port 443
exec uvicorn server:app --host 0.0.0.0 --port 443 --workers 1
SH
chmod +x "$ROOT/start.sh"

echo "✅ Zbudowano frontend+backend. Start: sudo /workspace/mrd69/start.sh"
