#!/bin/bash

# Przejdź do katalogu roboczego, jeśli skrypt jest uruchamiany z innego miejsca
cd "$(dirname "$0")" || exit

echo "--- 🚀 Konfiguracja środowiska wirtualnego ---"
if [ ! -d ".venv" ]; then
    echo "Tworzenie .venv..."
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "--- 📦 Instalacja zależności ---"
pip install --upgrade pip > /dev/null
echo "Instalowanie zależności z pyproject.toml..."
pip install .

echo "Instalowanie dodatkowych pakietów (LangChain, AI, Pamięć)..."
pip install langchain langchain-openai openai sentence-transformers scikit-learn networkx uvicorn

echo "--- 💾 Przygotowanie danych ---"
mkdir -p data
touch data/memory.jsonl
echo "Utworzono pusty plik data/memory.jsonl (jeśli nie istniał)."

echo "--- 🌐 Tworzenie plików frontendu ---"
mkdir -p static

cat > static/index.html << 'EOF'
<!DOCTYPE html><html><head><title>Mordzix Server</title></head>
<body style="font-family:Arial;background:#1a1a1a;color:white;text-align:center;padding-top:50px;">
<h1>🧠 Mordzix Działa</h1><p><a href="/mordzix" style="color:#0af;">Przejdź do Chatu</a></p>
</body></html>
EOF

cat > static/mordzix.html << 'EOF'
<!DOCTYPE html><html><head><title>Mordzix Chat</title><style>body{font-family:Arial;background:#111;color:#eee;display:flex;flex-direction:column;height:95vh;margin:0;padding:10px;}#chat{flex-grow:1;overflow-y:auto;border:1px solid #444;padding:10px;margin-bottom:10px;}#input-area{display:flex;gap:10px;}input{flex-grow:1;padding:10px;background:#333;color:#eee;border:1px solid #555;}button{padding:10px 20px;background:#06c;color:white;border:none;cursor:pointer;}</style></head><body><div id="chat"></div><div id="input-area"><input id="input" placeholder="Napisz..."><button id="send">Wyślij</button></div><script>const chat=document.getElementById("chat"),input=document.getElementById("input"),send=document.getElementById("send");const add=(m,s)=>{const p=document.createElement("p");p.innerHTML=`<strong>${s}:</strong> ${m}`;chat.appendChild(p);chat.scrollTop=chat.scrollHeight;};send.onclick=async()=>{const m=input.value;if(!m)return;add(m,"Ty");input.value="";try{const r=await fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:m,mode:"mordzix"})}),d=await r.json();add(d.response||"Błąd","Mordzix")}catch(e){add("Błąd połączenia: "+e.message,"System")}};input.onkeypress=e=>{if(e.key==="Enter")send.click()};add("System pamięci gotowy.","Mordzix");</script></body></html>
EOF

echo "--- 🛑 Zatrzymywanie starych procesów serwera ---"
pkill -f uvicorn || true
sleep 2

echo "--- ▶️ Uruchamianie serwera Mordzix ---"
echo "Serwer będzie dostępny pod adresem http://<TWÓJ_IP_RUNPOD>:5959"
python3 -m uvicorn main:app --host 0.0.0.0 --port 5959
