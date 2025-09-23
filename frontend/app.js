(() => {
  const els = {
    app: document.getElementById('app'),
    drawer: document.getElementById('drawer'),
    openDrawer: document.getElementById('openDrawer'),
    closeDrawer: document.getElementById('closeDrawer'),
    newChat: document.getElementById('newChat'),
    reloadBootstrap: document.getElementById('reloadBootstrap'),
    historyList: document.getElementById('historyList'),
    historySearch: document.getElementById('historySearch'),
    status: document.getElementById('status'),
    messages: document.getElementById('messages'),
    typing: document.getElementById('typing'),
    btnAttach: document.getElementById('btnAttach'),
    btnClearAttach: document.getElementById('btnClearAttach'),
    fileInput: document.getElementById('fileInput'),
    attachPreview: document.getElementById('attachPreview'),
    btnMic: document.getElementById('btnMic'),
    textInput: document.getElementById('textInput'),
    btnSend: document.getElementById('btnSend'),
  };

  const API = { base:'/api',
    bootstrap:'/bootstrap', history:'/history', historyLoad:id=>`/history/${id}`, chat:'/chat', upload:'/upload'
  };

  const S = { online:false, chatId:null, messages:[], pending:[], recognition:null, recognizing:false, history:[], bootstrap:null };

  const now=()=>new Date();
  const fmt=d=>d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
  const esc=s=>s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const id=()=>Math.random().toString(36).slice(2);

  function setOnline(v){ S.online=v; els.status.textContent=v?'Online':'Offline';
    els.status.classList.toggle('online',v); els.status.classList.toggle('offline',!v); }
  function scrollBottom(){ requestAnimationFrame(()=>{ els.messages.scrollTop = els.messages.scrollHeight + 999; }); }

  function render(){
    els.messages.innerHTML='';
    let prev=null;
    S.messages.forEach(m=>{
      if(prev && prev!==m.role){ const g=document.createElement('div'); g.className='group-gap'; els.messages.appendChild(g); }
      const b=document.createElement('div');
      b.className=`msg ${m.role==='user'?'me tail-right':'ai tail-left'}`;
      b.innerHTML=esc(m.content);
      const t=document.createElement('div'); t.className='time'; t.textContent=fmt(now()); b.appendChild(t);
      els.messages.appendChild(b); prev=m.role;
    });
    scrollBottom();
  }
  function push(role,content){ S.messages.push({role,content}); render(); }
  function typing(on){ els.typing.classList.toggle('hidden',!on); els.typing.setAttribute('aria-hidden',String(!on)); scrollBottom(); }

  // Drawer
  const openDrawer=()=>els.app.classList.add('app--drawer-open');
  const closeDrawer=()=>els.app.classList.remove('app--drawer-open');
  document.getElementById('openDrawer')?.addEventListener('click',openDrawer);
  document.getElementById('closeDrawer')?.addEventListener('click',closeDrawer);

  // Attachments
  els.btnAttach.addEventListener('click',()=>els.fileInput.click());
  els.fileInput.addEventListener('change',e=>{
    const files=Array.from(e.target.files||[]);
    for(const f of files) S.pending.push({tempId:id(),name:f.name,size:f.size,file:f});
    renderPending(); els.textInput.focus();
  });
  els.btnClearAttach.addEventListener('click',()=>{ S.pending=[]; renderPending(); });
  function renderPending(){
    els.attachPreview.innerHTML='';
    S.pending.forEach(a=>{
      const chip=document.createElement('span'); chip.className='attach-chip';
      chip.innerHTML=`${esc(a.name)} <button aria-label="Usuń">×</button>`;
      chip.querySelector('button').addEventListener('click',()=>{ S.pending=S.pending.filter(x=>x.tempId!==a.tempId); renderPending(); });
      els.attachPreview.appendChild(chip);
    });
  }

  // Speech-to-text
  function initSTT(){
    const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
    if(!SR){ els.btnMic.addEventListener('click',()=>toast('Brak natywnego STT.')); return; }
    const r=new SR(); r.lang='pl-PL'; r.continuous=true; r.interimResults=true; r.maxAlternatives=1;
    r.onstart=()=>{ S.recognizing=true; els.btnMic.classList.add('active'); };
    r.onend=()=>{ S.recognizing=false; els.btnMic.classList.remove('active'); };
    r.onerror=()=>toast('Błąd rozpoznawania mowy.');
    r.onresult=ev=>{
      let interim='', final=els.textInput.value||'';
      for(let i=ev.resultIndex;i<ev.results.length;i++){ const rr=ev.results[i]; if(rr.isFinal) final+=rr[0].transcript; else interim+=rr[0].transcript; }
      els.textInput.value=(final+(interim?' '+interim:'')); els.textInput.style.opacity=interim?'0.9':'1';
    };
    S.recognition=r;
    els.btnMic.addEventListener('click',()=>{ if(S.recognizing) r.stop(); else { try{ r.start(); }catch{} } });
  }

  // Bootstrap + historia
  async function boot(){
    try{ const r=await fetch(API.base+API.bootstrap); if(!r.ok) throw 0; S.bootstrap=await r.json(); setOnline(true); }
    catch{ S.bootstrap={version:'local'}; setOnline(false); }
  }
  async function loadHistory(){
    try{ const r=await fetch(API.base+API.history); if(r.ok){ S.history=await r.json()||[]; drawHistory(); return; } } catch{}
    const local=JSON.parse(localStorage.getItem('hist')||'[]'); S.history=local; drawHistory();
  }
  function drawHistory(filter=''){
    els.historyList.innerHTML='';
    const items=!filter?S.history:S.history.filter(h=>(h.title||'').toLowerCase().includes(filter));
    if(!items.length){ const e=document.createElement('div'); e.style.opacity='.7'; e.textContent='Brak rozmów'; els.historyList.appendChild(e); return; }
    for(const h of items){
      const div=document.createElement('div'); div.className='item'; div.textContent=h.title||new Date(h.ts||Date.now()).toLocaleString();
      div.addEventListener('click',()=>{ openChat(h.id).then(closeDrawer); }); els.historyList.appendChild(div);
    }
  }
  async function openChat(id){
    try{ const r=await fetch(API.base+API.historyLoad(id)); if(r.ok){ const d=await r.json(); S.chatId=id; S.messages=d.messages||[]; render(); return; } } catch{}
    const local=JSON.parse(localStorage.getItem('chat_'+id)||'{"messages":[]}'); S.chatId=id; S.messages=local.messages||[]; render();
  }
  function snapshot(){
    const list=JSON.parse(localStorage.getItem('hist')||'[]'); const item={id:S.chatId,title:S.messages[0]?.content?.slice(0,40)||'Rozmowa',ts:Date.now()};
    const i=list.findIndex(x=>x.id===S.chatId); if(i>=0) list[i]=item; else list.unshift(item);
    localStorage.setItem('hist',JSON.stringify(list.slice(0,100))); localStorage.setItem('chat_'+S.chatId,JSON.stringify({messages:S.messages}));
  }
  function newChat(){
    S.chatId='local-'+Date.now(); S.messages=[]; render(); if(S.bootstrap){ push('assistant','Wczytano pakiet ('+(S.bootstrap.version||'local')+').'); }
  }

  // Wysyłanie
  els.btnSend.addEventListener('click',send);
  els.textInput.addEventListener('keydown',e=>{ if(e.key==='Enter'){ e.preventDefault(); send(); } });
  async function send(){
    const text=els.textInput.value.trim();
    if(!text && !S.pending.length) return;
    if(S.pending.length && !text){ toast('Dodano pliki — napisz krótki opis.'); return; }
    push('user',text||'(bez treści)'); els.textInput.value=''; typing(true);

    // upload
    let uploaded=[];
    if(S.pending.length){
      if(S.online){
        for(const a of S.pending){
          try{ const fd=new FormData(); fd.append('file',a.file,a.name);
               const r=await fetch(API.base+API.upload,{method:'POST',body:fd});
               uploaded.push(r.ok?await r.json():{name:a.name,local:true}); }
          catch{ uploaded.push({name:a.name,local:true}); }
        }
      } else { uploaded=S.pending.map(a=>({name:a.name,local:true})); }
      S.pending=[]; renderPending();
    }

    // request
    let reply='';
    if(S.online){
      try{
        const r=await fetch(API.base+API.chat,{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({chatId:S.chatId,messages:S.messages,newMessage:{role:'user',content:text||'(bez treści)'},
                               attachments:uploaded,lang:'pl-PL'})});
        reply=r.ok?(await r.json()).reply||'(pusto)':'(błąd backendu)';
      }catch{ reply='(backend niedostępny)'; }
    }else reply=`Tryb offline: echo → ${text}`;

    typing(false); push('assistant',reply); snapshot();
  }

  // Toast (nie blokuje klików)
  let toastTimer=null;
  function toast(msg){
    let t=document.getElementById('toast'); if(!t){ t=document.createElement('div'); t.id='toast'; document.body.appendChild(t); }
    t.textContent=msg; t.style.opacity='1'; clearTimeout(toastTimer); toastTimer=setTimeout(()=>t.style.opacity='0',1600);
  }

  // UX
  els.historySearch.addEventListener('input',()=>drawHistory(els.historySearch.value.trim().toLowerCase()));
  document.getElementById('linkDocs').addEventListener('click',e=>{e.preventDefault();toast('Podmień link.');});
  document.getElementById('linkProject').addEventListener('click',e=>{e.preventDefault();toast('Podłącz panel pamięci.');});
  document.getElementById('linkSettings').addEventListener('click',e=>{e.preventDefault();toast('Ustawienia — TODO.');});
  document.getElementById('newChat').addEventListener('click',()=>{ newChat(); closeDrawer(); });
  document.getElementById('reloadBootstrap').addEventListener('click',async()=>{ await boot(); toast('Wiedza przeładowana'); });

  window.addEventListener('focusin',e=>{ if(e.target===els.textInput) scrollBottom(); });

  (async function init(){ initSTT(); await boot(); await loadHistory(); newChat(); })();
})();
