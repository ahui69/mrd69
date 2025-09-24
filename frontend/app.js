(() => {
  const API_BASE = '/api';

  const $ = (sel, parent = document) => parent.querySelector(sel);
  const $$ = (sel, parent = document) => Array.from(parent.querySelectorAll(sel));
  const toastEl = $('#toast');
  const globalStatus = $('#globalStatus');

  const navButtons = $$('.nav__btn');
  const sections = $$('.section');

  let activeSection = 'chat';

  function showToast(message, duration = 2200) {
    toastEl.textContent = message;
    toastEl.style.opacity = '1';
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(() => { toastEl.style.opacity = '0'; }, duration);
  }

  function switchSection(section) {
    activeSection = section;
    navButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.section === section));
    sections.forEach(sec => sec.classList.toggle('section--active', sec.id === `section-${section}`));
  }

  navButtons.forEach(btn => btn.addEventListener('click', () => switchSection(btn.dataset.section)));

  async function request(path, { method = 'GET', data, headers, raw } = {}) {
    const options = { method, headers: headers || {} };
    if (data !== undefined) {
      options.headers['Content-Type'] = 'application/json';
      options.body = typeof data === 'string' ? data : JSON.stringify(data);
    }
    const res = await fetch(`${API_BASE}${path}`, options);
    if (!res.ok) {
      const message = await res.text().catch(() => res.statusText);
      throw new Error(message || `${res.status}`);
    }
    if (raw) return res;
    const text = await res.text();
    return text ? JSON.parse(text) : null;
  }

  function renderJSON(target, payload, label) {
    const el = typeof target === 'string' ? document.getElementById(target) : target;
    if (!el) return;
    el.innerHTML = '';
    const pre = document.createElement('pre');
    pre.textContent = payload ? JSON.stringify(payload, null, 2) : 'brak danych';
    if (label) {
      const h = document.createElement('h4');
      h.textContent = label;
      el.appendChild(h);
    }
    el.appendChild(pre);
  }

  // ----------------------------- CHAT ---------------------------------
  const chatEls = {
    status: globalStatus,
    list: $('#chatHistoryList'),
    search: $('#chatHistorySearch'),
    messages: $('#chatMessages'),
    typing: $('#chatTyping'),
    form: $('#chatForm'),
    input: $('#chatInput'),
    refresh: $('#chatRefresh'),
    newChat: $('#chatNew')
  };

  const chatState = {
    online: false,
    bootstrapVersion: 'local',
    history: [],
    filter: '',
    chatId: null,
    messages: []
  };

  function setChatStatus(online, note) {
    chatState.online = online;
    chatEls.status.textContent = online ? `Online${note ? ' · ' + note : ''}` : 'Offline';
    chatEls.status.classList.toggle('active', online);
  }

  function renderChatMessages() {
    const { messages } = chatState;
    chatEls.messages.innerHTML = '';
    messages.forEach(msg => {
      const div = document.createElement('div');
      div.className = `msg ${msg.role === 'user' ? 'me' : 'ai'}`;
      div.innerHTML = `${msg.content}`;
      const time = document.createElement('div');
      time.className = 'time';
      time.textContent = new Date(msg.ts || Date.now()).toLocaleTimeString('pl-PL', { hour: '2-digit', minute: '2-digit' });
      div.appendChild(time);
      chatEls.messages.appendChild(div);
    });
    chatEls.messages.scrollTop = chatEls.messages.scrollHeight + 1000;
  }

  function renderChatHistory() {
    const list = chatEls.list;
    list.innerHTML = '';
    const term = chatState.filter.trim().toLowerCase();
    const items = !term
      ? chatState.history
      : chatState.history.filter(item => (item.title || '').toLowerCase().includes(term));
    if (!items.length) {
      const empty = document.createElement('div');
      empty.style.opacity = '0.65';
      empty.textContent = 'Brak rozmów';
      list.appendChild(empty);
      return;
    }
    items.forEach(item => {
      const div = document.createElement('div');
      div.className = 'item';
      div.textContent = item.title || new Date(item.ts || Date.now()).toLocaleString('pl-PL');
      if (item.id === chatState.chatId) div.classList.add('active');
      div.addEventListener('click', () => openChat(item.id));
      list.appendChild(div);
    });
  }

  async function loadBootstrap() {
    try {
      const health = await request('/health');
      setChatStatus(true, health?.mode || 'llm');
      chatState.bootstrapVersion = health?.version || 'remote';
    } catch (err) {
      setChatStatus(false);
      showToast('Backend /api/health niedostêpny');
    }
  }

  async function loadHistory() {
    try {
      const history = await request('/history');
      chatState.history = history || [];
      renderChatHistory();
    } catch (err) {
      chatState.history = [];
      renderChatHistory();
      showToast('Nie uda³o siê pobraæ historii');
    }
  }

  async function openChat(id) {
    try {
      const data = await request(`/history/${encodeURIComponent(id)}`);
      chatState.chatId = id;
      chatState.messages = (data?.messages || []).map(m => ({ ...m }));
      renderChatHistory();
      renderChatMessages();
    } catch (err) {
      showToast('Nie uda³o siê wczytaæ rozmowy');
    }
  }

  async function startNewChat() {
    chatState.chatId = `local-${Date.now()}`;
    chatState.messages = [];
    renderChatHistory();
    renderChatMessages();
  }

  async function sendMessage(event) {
    event.preventDefault();
    const text = chatEls.input.value.trim();
    if (!text) return;
    const userMsg = { role: 'user', content: text, ts: Date.now() };
    chatState.messages.push(userMsg);
    chatEls.input.value = '';
    renderChatMessages();
    chatEls.typing.classList.remove('hidden');

    const payload = {
      chatId: chatState.chatId,
      messages: chatState.messages,
      newMessage: { role: 'user', content: text },
      attachments: [],
      lang: 'pl-PL'
    };

    try {
      const res = await request('/chat', { method: 'POST', data: payload });
      if (!chatState.chatId) chatState.chatId = res?.chatId || `local-${Date.now()}`;
      const reply = { role: 'assistant', content: res?.reply || '(pusto)', ts: Date.now() };
      chatState.messages.push(reply);
      renderChatMessages();
      await loadHistory();
    } catch (err) {
      showToast(`B³¹d czatu: ${err.message}`);
      const fallback = { role: 'assistant', content: '(b³¹d backendu)', ts: Date.now() };
      chatState.messages.push(fallback);
      renderChatMessages();
    } finally {
      chatEls.typing.classList.add('hidden');
    }
  }

  // --------------------------- LISTINGS --------------------------------
  const listingsOutput = $('#listingsOutput');
  const listingsCreateForm = $('#listingsCreateForm');
  const listingsSearchForm = $('#listingsSearchForm');

  async function handleListingsCreate(ev) {
    ev.preventDefault();
    const form = new FormData(listingsCreateForm);
    const payload = Object.fromEntries(form.entries());
    Object.keys(payload).forEach(key => {
      if (payload[key] === '') delete payload[key];
    });
    ['base_price', 'profanity_level'].forEach(key => {
      if (payload[key] !== undefined && payload[key] !== '') payload[key] = Number(payload[key]);
    });
    try {
      const res = await request('/listings/create', { method: 'POST', data: payload });
      renderJSON(listingsOutput, res, 'Nowy listing');
      showToast('Listing utworzony');
    } catch (err) {
      renderJSON(listingsOutput, { error: err.message });
      showToast('B³¹d tworzenia listingów');
    }
  }

  async function handleListingsSearch(ev) {
    ev.preventDefault();
    const form = new FormData(listingsSearchForm);
    const params = new URLSearchParams();
    form.forEach((value, key) => {
      if (value) params.append(key, value.toString());
    });
    try {
      const res = await request(`/listings/search?${params.toString()}`);
      renderJSON(listingsOutput, res, 'Wyniki wyszukiwania');
      showToast(`Znaleziono ${res?.count ?? 0}`);
    } catch (err) {
      renderJSON(listingsOutput, { error: err.message });
      showToast('B³¹d wyszukiwania');
    }
  }

  async function listingsAction(kind) {
    const form = new FormData(listingsSearchForm);
    const slug = form.get('slug') || '';
    let path = '';
    if (kind === 'health') path = '/listings/health';
    if (kind === 'metrics') path = `/listings/metrics${slug ? `?slug=${encodeURIComponent(slug)}` : ''}`;
    if (kind === 'prices') path = `/listings/dynamic-prices${slug ? `?slug=${encodeURIComponent(slug)}` : ''}`;
    if (!path) return;
    try {
      const res = await request(path);
      renderJSON(listingsOutput, res, kind === 'health' ? 'Stan serwisu' : 'Dane');
      showToast('Pobrano dane listingów');
    } catch (err) {
      renderJSON(listingsOutput, { error: err.message });
      showToast('B³¹d listingu');
    }
  }

  // --------------------------- TRAVEL ----------------------------------
  const travelForm = $('#travelPlanForm');
  const travelOutput = $('#travelOutput');

  function getTravelFormData() {
    const form = new FormData(travelForm);
    const data = Object.fromEntries(form.entries());
    data.days = Number(data.days || 1);
    data.food = form.get('food') !== null;
    data.hotels = form.get('hotels') !== null;
    return data;
  }

  async function handleTravelPlan(ev) {
    ev.preventDefault();
    const data = getTravelFormData();
    const payload = {
      place: data.place,
      days: data.days,
      food: data.food,
      hotels: data.hotels,
      lang: 'pl',
      user_id: data.user || 'default',
      narrative_style: data.style || 'auto'
    };
    try {
      const res = await request('/travel/plan', { method: 'POST', data: payload });
      renderJSON(travelOutput, res, 'Plan podró¿y');
      showToast('Plan gotowy');
    } catch (err) {
      renderJSON(travelOutput, { error: err.message });
      showToast('B³¹d planowania');
    }
  }

  async function travelAction(kind) {
    const data = getTravelFormData();
    try {
      let res;
      switch (kind) {
        case 'restaurants':
          res = await request(`/travel/restaurants?place=${encodeURIComponent(data.place)}&max_results=20&lang=pl`);
          renderJSON(travelOutput, res, 'Restauracje');
          break;
        case 'hotels':
          res = await request(`/travel/hotels?place=${encodeURIComponent(data.place)}&max_results=20&lang=pl`);
          renderJSON(travelOutput, res, 'Hotele');
          break;
        case 'attractions':
          res = await request(`/travel/attractions?place=${encodeURIComponent(data.place)}&max_results=40&lang=pl`);
          renderJSON(travelOutput, res, 'Atrakcje');
          break;
        case 'flights': {
          const origin = data.origin || 'WAW';
          const destination = data.destination || data.place || 'BCN';
          const month = data.month || new Date().toISOString().slice(0, 7);
          res = await request(`/travel/flights?origin=${encodeURIComponent(origin)}&dest=${encodeURIComponent(destination)}&month=${encodeURIComponent(month)}`);
          renderJSON(travelOutput, res, 'Loty');
          break; }
        case 'transit':
          res = await request(`/travel/transit?place=${encodeURIComponent(data.place)}&radius=900`);
          renderJSON(travelOutput, res, 'Transport publiczny');
          break;
        case 'map':
          res = await request(`/travel/map?place=${encodeURIComponent(data.place)}&max_items=40`);
          renderJSON(travelOutput, res, 'Mapa i pinezki');
          break;
        default:
          return;
      }
      showToast('Dane podró¿y zaktualizowane');
    } catch (err) {
      renderJSON(travelOutput, { error: err.message });
      showToast('B³¹d travel');
    }
  }

  // --------------------------- CRYPTO ----------------------------------
  const cryptoOutput = $('#cryptoOutput');
  const cryptoTokenForm = $('#cryptoTokenForm');
  const cryptoPortfolioForm = $('#cryptoPortfolioForm');

  async function cryptoHealth() {
    try {
      const res = await request('/crypto/health');
      renderJSON(cryptoOutput, res, 'Stan serwisu');
      showToast('Crypto API dzia³a');
    } catch (err) {
      renderJSON(cryptoOutput, { error: err.message });
      showToast('B³¹d /api/crypto');
    }
  }

  async function handleCryptoToken(ev) {
    ev.preventDefault();
    const form = new FormData(cryptoTokenForm);
    const token = form.get('token');
    const vs = form.get('vs') || 'usd';
    try {
      const res = await request(`/crypto/token/${encodeURIComponent(token)}?vs=${encodeURIComponent(vs)}`);
      renderJSON(cryptoOutput, res, `Token ${token}`);
      showToast('Analiza tokena gotowa');
    } catch (err) {
      renderJSON(cryptoOutput, { error: err.message });
      showToast('B³¹d analizy tokena');
    }
  }

  async function cryptoTokenAction(kind) {
    const form = new FormData(cryptoTokenForm);
    const token = form.get('token');
    const vs = form.get('vs') || 'usd';
    try {
      let res;
      if (kind === 'report') {
        res = await request(`/crypto/token/${encodeURIComponent(token)}/report?horizon=30d&risk=medium`);
        renderJSON(cryptoOutput, res, 'Raport tokena');
      } else if (kind === 'score') {
        res = await request(`/crypto/token/${encodeURIComponent(token)}/score?vs=${encodeURIComponent(vs)}`);
        renderJSON(cryptoOutput, res, 'Ocena tokena');
      } else if (kind === 'backtest') {
        res = await request(`/crypto/token/${encodeURIComponent(token)}/backtest?days=90&short=5&long=20&capital=1000`);
        renderJSON(cryptoOutput, res, 'Backtest SMA');
      } else if (kind === 'screener') {
        const limit = Number(form.get('limit') || 20);
        res = await request(`/crypto/screener?limit=${limit}&vs=${encodeURIComponent(vs)}`);
        renderJSON(cryptoOutput, res, 'Screener');
      }
      showToast('Zapytanie crypto OK');
    } catch (err) {
      renderJSON(cryptoOutput, { error: err.message });
      showToast('B³¹d crypto');
    }
  }

  async function cryptoPortfolioAction(kind) {
    const form = new FormData(cryptoPortfolioForm);
    const user = form.get('user') || 'demo';
    const address = (form.get('address') || '').toString();
    try {
      let res;
      if (kind === 'add') {
        if (!address) { showToast('Podaj adres ETH'); return; }
        res = await request(`/crypto/portfolio/${encodeURIComponent(user)}/addresses`, { method: 'POST', data: { address } });
        renderJSON(cryptoOutput, res, 'Dodano adres');
      } else if (kind === 'scan') {
        res = await request(`/crypto/portfolio/${encodeURIComponent(user)}/scan`, { method: 'POST' });
        renderJSON(cryptoOutput, res, 'Skan portfela');
      } else if (kind === 'report') {
        res = await request(`/crypto/portfolio/${encodeURIComponent(user)}/report`);
        renderJSON(cryptoOutput, res, 'Raport portfela');
      } else if (kind === 'screener') {
        await cryptoTokenAction('screener');
        return;
      }
      showToast('Operacja portfela OK');
    } catch (err) {
      renderJSON(cryptoOutput, { error: err.message });
      showToast('B³¹d portfela');
    }
  }

  // --------------------------- MEMORY ----------------------------------
  const memoryOutput = $('#memoryOutput');
  const memoryForm = $('#memoryForm');

  async function memoryAction(kind) {
    try {
      let res;
      if (kind === 'health') {
        res = await request('/memory/health');
        renderJSON(memoryOutput, res, 'Health');
      } else if (kind === 'stats') {
        res = await request('/memory/stats');
        renderJSON(memoryOutput, res, 'Statystyki');
      } else if (kind === 'recall') {
        const q = new FormData(memoryForm).get('query') || 'ostatnie fakty';
        res = await request('/memory/recall', { method: 'POST', data: { q, topk: 6 } });
        renderJSON(memoryOutput, res, `Recall: ${q}`);
      }
      showToast('Zapytanie pamiêci wykonane');
    } catch (err) {
      renderJSON(memoryOutput, { error: err.message });
      showToast('B³¹d pamiêci');
    }
  }

  // --------------------------- LISTENERY --------------------------------
  chatEls.form.addEventListener('submit', sendMessage);
  chatEls.refresh.addEventListener('click', async () => { await loadBootstrap(); await loadHistory(); showToast('Odœwie¿ono'); });
  chatEls.newChat.addEventListener('click', () => { startNewChat(); showToast('Nowy czat'); });
  chatEls.search.addEventListener('input', ev => { chatState.filter = ev.target.value; renderChatHistory(); });

  listingsCreateForm.addEventListener('submit', handleListingsCreate);
  listingsSearchForm.addEventListener('submit', handleListingsSearch);
  $$("[data-action='listings-health']").forEach(btn => btn.addEventListener('click', () => listingsAction('health')));
  $$("[data-action='listings-metrics']").forEach(btn => btn.addEventListener('click', () => listingsAction('metrics')));
  $$("[data-action='listings-prices']").forEach(btn => btn.addEventListener('click', () => listingsAction('prices')));

  travelForm.addEventListener('submit', handleTravelPlan);
  $$("[data-action='travel-restaurants']").forEach(btn => btn.addEventListener('click', () => travelAction('restaurants')));
  $$("[data-action='travel-hotels']").forEach(btn => btn.addEventListener('click', () => travelAction('hotels')));
  $$("[data-action='travel-attractions']").forEach(btn => btn.addEventListener('click', () => travelAction('attractions')));
  $$("[data-action='travel-flights']").forEach(btn => btn.addEventListener('click', () => travelAction('flights')));
  $$("[data-action='travel-transit']").forEach(btn => btn.addEventListener('click', () => travelAction('transit')));
  $$("[data-action='travel-map']").forEach(btn => btn.addEventListener('click', () => travelAction('map')));

  cryptoTokenForm.addEventListener('submit', handleCryptoToken);
  $$("[data-action='crypto-health']").forEach(btn => btn.addEventListener('click', cryptoHealth));
  $$("[data-action='crypto-report']").forEach(btn => btn.addEventListener('click', () => cryptoTokenAction('report')));
  $$("[data-action='crypto-score']").forEach(btn => btn.addEventListener('click', () => cryptoTokenAction('score')));
  $$("[data-action='crypto-backtest']").forEach(btn => btn.addEventListener('click', () => cryptoTokenAction('backtest')));
  $$("[data-action='crypto-screener']").forEach(btn => btn.addEventListener('click', () => cryptoTokenAction('screener')));
  $$("[data-action='crypto-add']").forEach(btn => btn.addEventListener('click', () => cryptoPortfolioAction('add')));
  $$("[data-action='crypto-scan']").forEach(btn => btn.addEventListener('click', () => cryptoPortfolioAction('scan')));
  $$("[data-action='crypto-portfolio-report']").forEach(btn => btn.addEventListener('click', () => cryptoPortfolioAction('report')));

  $$("[data-action='memory-health']").forEach(btn => btn.addEventListener('click', () => memoryAction('health')));
  $$("[data-action='memory-stats']").forEach(btn => btn.addEventListener('click', () => memoryAction('stats')));
  $$("[data-action='memory-recall']").forEach(btn => btn.addEventListener('click', () => memoryAction('recall')));

  // --------------------------- INIT -------------------------------------
  (async () => {
    switchSection('chat');
    await loadBootstrap();
    await loadHistory();
    await startNewChat();
    showToast('MRD69 gotowy');
  })();
})();
