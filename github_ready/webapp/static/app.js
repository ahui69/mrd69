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
