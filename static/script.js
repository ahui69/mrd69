document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const micButton = document.getElementById('mic-button');
    const fileInput = document.getElementById('file-input');
    const messagesContainer = document.getElementById('messages');
    const typingIndicator = document.getElementById('typing-indicator');
    const filePreviewArea = document.getElementById('file-preview-area');
    const btnNew = document.getElementById('btn-new');
    const btnHistory = document.getElementById('btn-history');
    const btnMaps = document.getElementById('btn-maps');
    const btnSearch = document.getElementById('btn-search');
    const historyPanel = document.getElementById('history-panel');
    const mapsPanel = document.getElementById('maps-panel');
    const searchPanel = document.getElementById('search-panel');
    const mapsCloseBtn = document.getElementById('maps-close');
    const searchCloseBtn = document.getElementById('search-close');
    const mapsGpsBtn = document.getElementById('maps-gps-btn');
    const mapsRouteBtn = document.getElementById('maps-route-btn');
    const searchBtn = document.getElementById('search-btn');
    const searchAutoCheck = document.getElementById('search-auto');
    const mapsStartInput = document.getElementById('maps-start');
    const mapsEndInput = document.getElementById('maps-end');
    const searchQueryInput = document.getElementById('search-query');
    const searchResults = document.getElementById('search-results');
    const mapsContainer = document.getElementById('maps-container');
    const routeInfo = document.getElementById('route-info');
    const routeSteps = document.getElementById('route-steps');
    const historyList = document.getElementById('history-list');
    const statusDot = document.getElementById('status-dot');
    const chatMessages = document.getElementById('chat-messages');
    const ttsToggle = document.getElementById('tts-toggle');
    const mapsMode = document.getElementById('maps-mode');
    const mapsSwapBtn = document.getElementById('maps-swap-btn');

    // State variables
    let socket;
    let recognition;
    let isRecording = false;
    const selectedFiles = [];
    // TTS state
    let ttsEnabled = false;
    let ttsVoice = null;
    let lastUtterance = null;
    
    // Auto resize input on typing
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Status indicator for connection
    function setStatus(state) {
        // state: 'connected' | 'connecting' | 'disconnected'
        if (!statusDot) return;
        if (state === 'connected') {
            statusDot.style.color = 'var(--color-online)';
            statusDot.title = 'Połączono';
        } else if (state === 'connecting') {
            statusDot.style.color = '#f1c40f';
            statusDot.title = 'Łączenie...';
        } else {
            statusDot.style.color = 'var(--color-offline)';
            statusDot.title = 'Rozłączono';
        }
    }

    // WebSocket connection handling
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        socket = new WebSocket(`${protocol}//${window.location.host}/ws`);

        socket.onopen = () => {
            console.log('WebSocket connected');
            setStatus('connected');
        };

        socket.onmessage = (event) => {
            typingIndicator.style.display = 'none';
            addMessage(event.data, 'assistant');
            scrollToBottom();
        };

        socket.onclose = () => {
            console.log('WebSocket disconnected. Reconnecting...');
            setStatus('disconnected');
            setTimeout(connectWebSocket, 1000); // Próba ponownego połączenia po 1s
        };

        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            setStatus('disconnected');
            socket.close();
        };

        setStatus('connecting');
    }

    // Add message to chat
    function addMessage(text, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        messageElement.textContent = text;
        messagesContainer.appendChild(messageElement); // Dodajemy na końcu - normalna kolejność
        
        // Escape HTML to prevent XSS
        const originalText = text;
        text = text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
        
        // Auto-linkify URLs
        if (sender !== 'system') {
            text = text.replace(
                /(\b(https?|ftp|file):\/\/[-A-Z0-9+&@#\/%?=~_|!:,.;]*[-A-Z0-9+&@#\/%=~_|])/ig, 
                '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
            );
        }
        
        messageElement.innerHTML = text;

        // Speak assistant replies if TTS is enabled
        if (sender === 'assistant') {
            speak(originalText);
        }
    }
    
    // Scroll to bottom of chat
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Clear chat
    function clearChat() {
        messagesContainer.innerHTML = '';
        typingIndicator.style.display = 'none';
        // wyczyść ewentualne pliki w kolejce
        selectedFiles.length = 0;
        filePreviewArea.innerHTML = '';
        fileInput.value = '';
    }

    // Send message function
    function sendMessage() {
        const messageText = messageInput.value.trim();
        if (messageText === '' && selectedFiles.length === 0) return;

        // Handle text message
        if (messageText !== '') {
            addMessage(messageText, 'user');
            scrollToBottom();
            
            try {
                if (socket && socket.readyState === WebSocket.OPEN) {
                    socket.send(messageText);
                    typingIndicator.style.display = 'flex';
                    scrollToBottom();
                } else {
                    addMessage('Brak połączenia z serwerem. Spróbuj ponownie.', 'system');
                }
            } catch (e) {
                addMessage('Błąd wysyłania wiadomości.', 'system');
            }
        }
        
        // Handle file uploads
        if (selectedFiles.length > 0) {
            const formData = new FormData();
            selectedFiles.forEach(file => {
                formData.append('file', file);
            });

            // System message about file processing
            addMessage('Przetwarzanie plików...', 'system');
            
            fetch('/uploadfile/', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.msg) {
                    addMessage(data.msg, 'system');
                } else if (data.error) {
                    addMessage(`Błąd wysyłania pliku: ${data.error}`, 'system');
                } else if (data.results && Array.isArray(data.results)) {
                    data.results.forEach(r => {
                        if (r.msg) addMessage(r.msg, 'system');
                        if (r.error) addMessage(`Błąd: ${r.error}`, 'system');
                    });
                }
                scrollToBottom();
            })
            .catch(error => {
                addMessage(`Błąd sieci: ${error}`, 'system');
                scrollToBottom();
            });

            // Reset file selection
            selectedFiles.length = 0;
            filePreviewArea.innerHTML = '';
            fileInput.value = '';
        }

        // Reset input area
        messageInput.value = '';
        messageInput.style.height = 'auto';
        messageInput.focus();
    }

    // Initialize chat with boot message
    function boot() {
        fetch('/boot')
            .then(r => r.json())
            .then(d => {
                if (d && d.msg) {
                    addMessage(d.msg, 'system');
                    scrollToBottom();
                }
            })
            .catch(() => {
                addMessage('Nie udało się załadować pakietu wiedzy.', 'system');
            });
    }

    // Load conversation history
    function loadHistory() {
        if (!historyList) return;
        
        // Clear current history list
        historyList.innerHTML = '';
        
        // Add loading indicator
        const loadingItem = document.createElement('li');
        loadingItem.textContent = 'Ładowanie historii...';
        loadingItem.classList.add('history-loading');
        historyList.appendChild(loadingItem);
        
        fetch('/episodes?limit=50')
            .then(r => r.json())
            .then(items => {
                // Remove loading indicator
                historyList.innerHTML = '';
                
                if (!Array.isArray(items) || items.length === 0) {
                    const emptyItem = document.createElement('li');
                    emptyItem.textContent = 'Brak historii rozmów';
                    emptyItem.classList.add('history-empty');
                    historyList.appendChild(emptyItem);
                    return;
                }
                
                items.forEach(ep => {
                    const li = document.createElement('li');
                    const u = (ep && ep.u) ? String(ep.u) : '';
                    const a = (ep && ep.a) ? String(ep.a) : '';
                    const ts = (ep && ep.ts) ? ep.ts : null;
                    
                    // Create preview text (max 30 chars)
                    const uPreview = u.length > 30 ? u.slice(0, 30) + '…' : u;
                    const aPreview = a.length > 30 ? a.slice(0, 30) + '…' : a;
                    
                    // Format date if available
                    const dateStr = ts ? new Date(ts * 1000).toLocaleString('pl-PL', {
                        day: 'numeric',
                        month: 'short',
                        hour: '2-digit',
                        minute: '2-digit'
                    }) : '';
                    
                    // Create HTML structure for history item
                    li.innerHTML = `
                        <div class="history-item-time">${dateStr}</div>
                        <div class="history-item-content">
                            ${uPreview ? `<div class="history-preview user">${uPreview}</div>` : ''}
                            ${aPreview ? `<div class="history-preview assistant">${aPreview}</div>` : ''}
                        </div>
                    `;
                    
                    if (ts) li.title = new Date(ts * 1000).toLocaleString('pl-PL');
                    li.addEventListener('click', () => restoreConversation(u, a));
                    historyList.appendChild(li);
                });
            })
            .catch(() => {
                historyList.innerHTML = '';
                const errorItem = document.createElement('li');
                errorItem.textContent = 'Błąd ładowania historii';
                errorItem.classList.add('history-error');
                historyList.appendChild(errorItem);
            });
    }

    // Toggle history panel visibility
    function toggleHistoryPanel() {
        if (!historyPanel) return;
        
        if (historyPanel.classList.contains('active')) {
            historyPanel.classList.remove('active');
        } else {
            historyPanel.classList.add('active');
            // Ukryj panel map, jeśli jest otwarty
            if (mapsPanel && mapsPanel.classList.contains('active')) {
                mapsPanel.classList.remove('active');
            }
            loadHistory();
        }
    }
    
    // Mapa i funkcjonalność wyznaczania tras
    let map, routeLayer, marker1, marker2;
    
    // Inicjalizacja mapy
    function initMap() {
        if (!mapsContainer || map) return;
        
        // Inicjalizacja mapy z centrum w Polsce
        map = L.map(mapsContainer).setView([52.0977, 19.0258], 6);
        
        // Dodanie warstwy OpenStreetMap
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            maxZoom: 19
        }).addTo(map);
        
        // Warstwa dla trasy
        routeLayer = L.layerGroup().addTo(map);
        
        // Odśwież mapę po załadowaniu panelu
        setTimeout(() => {
            map.invalidateSize();
        }, 400);
    }
    
    // Pokaż/ukryj panel map
    function toggleMapsPanel() {
        if (!mapsPanel) return;
        
        if (mapsPanel.classList.contains('active')) {
            mapsPanel.classList.remove('active');
        } else {
            mapsPanel.classList.add('active');
            // Ukryj panel historii, jeśli jest otwarty
            if (historyPanel && historyPanel.classList.contains('active')) {
                historyPanel.classList.remove('active');
            }
            
            // Inicjalizacja mapy przy pierwszym otwarciu
            if (!map) {
                initMap();
            } else {
                // Odśwież mapę po ponownym otwarciu
                setTimeout(() => {
                    map.invalidateSize();
                }, 400);
            }
        }
    }
    
    // Funkcja do wyznaczania trasy
    async function findRoute() {
        if (!map) return;

        const start = mapsStartInput.value.trim();
        const end = mapsEndInput.value.trim();

        if (!start || !end) {
            routeInfo.textContent = 'Podaj punkt startowy i cel podróży';
            return;
        }

    routeInfo.textContent = 'Wyznaczanie trasy...';
    if (routeSteps) routeSteps.innerHTML = '';

        // Wyczyść poprzednią trasę
        routeLayer.clearLayers();

        try {
            // Geokodowanie Nominatim
            const [startResults, endResults] = await Promise.all([
                fetch(`https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(start)}&format=json&limit=1`).then(r => r.json()),
                fetch(`https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(end)}&format=json&limit=1`).then(r => r.json())
            ]);

            if (!startResults.length || !endResults.length) {
                throw new Error('Nie można odnaleźć jednego z podanych miejsc');
            }

            const startLat = parseFloat(startResults[0].lat);
            const startLon = parseFloat(startResults[0].lon);
            const endLat = parseFloat(endResults[0].lat);
            const endLon = parseFloat(endResults[0].lon);

            const startCoords = [startLat, startLon];
            const endCoords = [endLat, endLon];

            const startName = (startResults[0].display_name || '').split(',')[0] || 'Start';
            const endName = (endResults[0].display_name || '').split(',')[0] || 'Cel';

            // Dodaj markery dla punktów
            if (marker1) map.removeLayer(marker1);
            if (marker2) map.removeLayer(marker2);

            marker1 = L.marker(startCoords, {
                icon: L.divIcon({
                    className: 'map-marker-start',
                    html: '<div class="marker-pin">A</div>',
                    iconSize: [30, 30],
                    iconAnchor: [15, 30]
                })
            }).addTo(map);

            marker2 = L.marker(endCoords, {
                icon: L.divIcon({
                    className: 'map-marker-end',
                    html: '<div class="marker-pin">B</div>',
                    iconSize: [30, 30],
                    iconAnchor: [15, 30]
                })
            }).addTo(map);

            // Dopasuj widok mapy do trasy
            const bounds = L.latLngBounds(startCoords, endCoords);
            map.fitBounds(bounds, { padding: [50, 50] });

            // OSRM bez klucza
            const profile = (mapsMode && mapsMode.value) ? mapsMode.value : 'driving';
            const url = `https://router.project-osrm.org/route/v1/${profile}/${startLon},${startLat};${endLon},${endLat}?overview=full&geometries=geojson&alternatives=false&steps=true&annotations=false`;
            const routeData = await fetch(url).then(r => r.json());

            if (!routeData || !routeData.routes || !routeData.routes.length) {
                throw new Error('Nie można wyznaczyć trasy między tymi punktami');
            }

            const route = routeData.routes[0];
            const coordinates = route.geometry.coordinates.map(coord => [coord[1], coord[0]]);

            L.polyline(coordinates, {
                color: '#4d4dff',
                weight: 5,
                opacity: 0.7
            }).addTo(routeLayer);

            const distance = (route.distance / 1000).toFixed(1); // km
            const durationMin = Math.round(route.duration / 60); // min

            routeInfo.innerHTML = `
                <strong>Trasa:</strong> ${startName} → ${endName}<br>
                <strong>Dystans:</strong> ${distance} km<br>
                <strong>Czas przejazdu:</strong> ${Math.floor(durationMin/60)}h ${durationMin%60}min
            `;

            // Render kroków nawigacji (pierwsza noga)
            if (routeSteps) {
                const steps = (route.legs && route.legs[0] && route.legs[0].steps) ? route.legs[0].steps : [];
                if (steps.length) {
                    const list = document.createElement('ol');
                    list.className = 'route-steps-list';

                    steps.forEach((s, idx) => {
                        const li = document.createElement('li');
                        li.className = 'route-step';
                        const dist = s.distance ? `${Math.round(s.distance)} m` : '';
                        const name = s.name || '';
                        const instr = toPolishInstruction(s.maneuver || {}, name);
                        li.innerHTML = `<span class="step-idx">${idx + 1}.</span> <span class="step-text">${instr}</span> <span class="step-dist">${dist}</span>`;
                        list.appendChild(li);
                    });
                    routeSteps.innerHTML = '';
                    routeSteps.appendChild(list);
                } else {
                    routeSteps.innerHTML = '<div class="route-steps-empty">Brak kroków nawigacji</div>';
                }
            }

            addMessage(`Wyznaczyłem trasę: ${startName} → ${endName}\nDystans: ${distance} km\nCzas przejazdu: około ${Math.floor(durationMin/60)}h ${durationMin%60}min`, 'system');
            scrollToBottom();
        } catch (error) {
            console.error('Błąd wyznaczania trasy:', error);
            routeInfo.textContent = (error && error.message) ? error.message : 'Wystąpił błąd podczas wyznaczania trasy';
            if (routeSteps) routeSteps.innerHTML = '';
        }
    }

    // Prosta konwersja instrukcji OSRM na polskie komendy
    function toPolishInstruction(maneuver, roadName) {
        const type = maneuver.type || '';
        const modifier = (maneuver.modifier || '').toLowerCase();
        const name = roadName ? ` na ${roadName}` : '';
        switch (type) {
            case 'depart':
                return 'Start';
            case 'arrive':
                return 'Cel po lewej/prawej stronie';
            case 'turn':
                if (modifier === 'left') return `Skręć w lewo${name}`;
                if (modifier === 'right') return `Skręć w prawo${name}`;
                if (modifier === 'slight left') return `Lekko w lewo${name}`;
                if (modifier === 'slight right') return `Lekko w prawo${name}`;
                if (modifier === 'sharp left') return `Ostro w lewo${name}`;
                if (modifier === 'sharp right') return `Ostro w prawo${name}`;
                return `Skręć${name}`;
            case 'merge':
                return `Włącz się do ruchu${name}`;
            case 'on ramp':
                return `Wjazd na ${name || 'zjazd'}`;
            case 'off ramp':
                return `Zjedź z drogi${name}`;
            case 'fork':
                if (modifier === 'left') return `Trzymaj się lewej${name}`;
                if (modifier === 'right') return `Trzymaj się prawej${name}`;
                return `Na rozwidleniu wybierz właściwy kierunek${name}`;
            case 'roundabout':
            case 'rotary':
                return 'Wjedź na rondo i zjedź odpowiednim zjazdem';
            case 'new name':
                return `Kontynuuj${name}`;
            case 'continue':
                return `Jedź prosto${name}`;
            case 'end of road':
                return `Koniec drogi, jedź zgodnie ze znakami${name}`;
            default:
                return `Jedź dalej${name}`;
        }
    }
    
    // Użyj aktualnej lokalizacji użytkownika
    function useCurrentLocation() {
        if (!navigator.geolocation) {
            routeInfo.textContent = 'Geolokalizacja nie jest wspierana przez twoją przeglądarkę';
            return;
        }
        
        routeInfo.textContent = 'Pobieranie aktualnej lokalizacji...';
        
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const lat = position.coords.latitude;
                const lng = position.coords.longitude;
                
                // Wykonaj odwrotne geokodowanie aby uzyskać nazwę miejsca
                fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lng}&format=json`)
                    .then(r => r.json())
                    .then(data => {
                        if (data.display_name) {
                            mapsStartInput.value = data.display_name;
                            routeInfo.textContent = 'Lokalizacja ustawiona jako punkt startowy';
                        } else {
                            mapsStartInput.value = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
                            routeInfo.textContent = 'Lokalizacja GPS ustawiona jako punkt startowy';
                        }
                    })
                    .catch(() => {
                        mapsStartInput.value = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
                        routeInfo.textContent = 'Lokalizacja GPS ustawiona jako punkt startowy';
                    });
            },
            (error) => {
                console.error('Błąd geolokalizacji:', error);
                routeInfo.textContent = 'Nie udało się pobrać lokalizacji. Sprawdź uprawnienia';
            },
            { enableHighAccuracy: true }
        );
    }

    // Restore conversation from history
    function restoreConversation(u, a) {
        fetch('/restore', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ u, a })
        })
        .then(r => r.json())
        .then(d => {
            if (d && d.ok) {
                clearChat();
                addMessage('Przywrócono kontekst rozmowy.', 'system');
                if (u) addMessage(u, 'user');
                if (a) addMessage(a, 'assistant');
                historyPanel.classList.remove('active');
                scrollToBottom();
            } else if (d && d.error) {
                addMessage(`Błąd przywracania: ${d.error}`, 'system');
            }
        })
        .catch(() => {
            addMessage('Błąd sieci przy przywracaniu rozmowy.', 'system');
        });
    }

    // Web Speech API for speech-to-text (Polish language support)
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        // Use standard SpeechRecognition if available, otherwise use webkit prefixed version
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'pl-PL'; // Polish language
        
        // Handle speech recognition results
        recognition.onresult = (event) => {
            let interim_transcript = '';
            let final_transcript = '';

            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    final_transcript += event.results[i][0].transcript;
                } else {
                    interim_transcript += event.results[i][0].transcript;
                }
            }
            
            // Update message input with recognized text
            messageInput.value = final_transcript + interim_transcript;
            // Auto resize the textarea to fit content
            messageInput.dispatchEvent(new Event('input'));
        };

        // Toggle recording state
        micButton.addEventListener('click', () => {
            if (isRecording) {
                recognition.stop();
                micButton.classList.remove('recording');
            } else {
                recognition.start();
                micButton.classList.add('recording');
            }
            isRecording = !isRecording;
        });

        // Handle recognition end
        recognition.onend = () => {
            // If still in recording mode but recognition ended, restart it
            if (isRecording) {
                recognition.start();
            } else {
                micButton.classList.remove('recording');
            }
        };
        
        // Handle errors
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            isRecording = false;
            micButton.classList.remove('recording');
            
            // Show error message if critical
            if (event.error === 'not-allowed') {
                addMessage('Dostęp do mikrofonu został zablokowany. Sprawdź ustawienia przeglądarki.', 'system');
            }
        };
    } else {
        // Hide mic button if Speech Recognition is not supported
        micButton.style.display = 'none';
        console.warn('Speech Recognition API is not supported in this browser');
    }

    // Speech Synthesis (TTS) for assistant responses
    function pickPolishVoice() {
        const voices = window.speechSynthesis ? window.speechSynthesis.getVoices() : [];
        if (!voices || voices.length === 0) return null;
        // Prefer Polish voices
        const plVoices = voices.filter(v => (v.lang || '').toLowerCase().startsWith('pl'));
        if (plVoices.length) return plVoices[0];
        // Fallback to any voice
        return voices[0] || null;
    }

    function refreshVoices() {
        if (!('speechSynthesis' in window)) return;
        ttsVoice = pickPolishVoice();
    }

    function speak(text) {
        if (!ttsEnabled || !('speechSynthesis' in window)) return;
        const clean = (text || '').toString();
        if (!clean) return;
        try {
            // Cancel any ongoing speech to avoid overlap
            window.speechSynthesis.cancel();
            const utter = new SpeechSynthesisUtterance(clean);
            if (ttsVoice) utter.voice = ttsVoice;
            utter.lang = (ttsVoice && ttsVoice.lang) ? ttsVoice.lang : 'pl-PL';
            utter.rate = 1.0;
            utter.pitch = 1.0;
            lastUtterance = utter;
            window.speechSynthesis.speak(utter);
        } catch (e) {
            console.warn('TTS speak failed:', e);
        }
    }

    if ('speechSynthesis' in window) {
        // Load voices now and when voiceschange fires
        refreshVoices();
        window.speechSynthesis.onvoiceschanged = refreshVoices;
    }

    if (ttsToggle) {
        ttsToggle.addEventListener('click', () => {
            ttsEnabled = !ttsEnabled;
            if (ttsEnabled) {
                ttsToggle.classList.add('active');
                ttsToggle.title = 'Głos asystenta: włączony';
                // Provide quick audible confirmation if possible
                speak('Głos asystenta włączony.');
            } else {
                ttsToggle.classList.remove('active');
                ttsToggle.title = 'Głos asystenta: wyłączony';
                if ('speechSynthesis' in window) window.speechSynthesis.cancel();
            }
        });
    }

    // Send message when send button is clicked
    sendButton.addEventListener('click', sendMessage);
    
    // Send message on Enter key (but allow multiline with Shift+Enter)
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // File input handling - process selected files
    fileInput.addEventListener('change', (event) => {
        // Clear previous selections if adding new files would exceed limit
        if (event.target.files.length + selectedFiles.length > 5) {
            // Show warning
            addMessage('Możesz wybrać maksymalnie 5 plików naraz.', 'system');
            
            // Keep only as many files as we can add
            const remainingSlots = Math.max(0, 5 - selectedFiles.length);
            
            for (let i = 0; i < Math.min(remainingSlots, event.target.files.length); i++) {
                selectedFiles.push(event.target.files[i]);
                displayFilePreview(event.target.files[i]);
            }
        } else {
            // Add all selected files
            for (const file of event.target.files) {
                selectedFiles.push(file);
                displayFilePreview(file);
            }
        }
    });

    // Display file preview in the preview area
    function displayFilePreview(file) {
        const reader = new FileReader();
        const filePreviewWrapper = document.createElement('div');
        filePreviewWrapper.classList.add('file-preview');

        // Create remove button for the file
        const removeButton = document.createElement('button');
        removeButton.classList.add('remove-file');
        removeButton.innerHTML = '&times;';
        removeButton.onclick = () => {
            const index = selectedFiles.indexOf(file);
            if (index > -1) {
                selectedFiles.splice(index, 1);
            }
            filePreviewWrapper.remove();
            
            // Hide preview area if empty
            if (selectedFiles.length === 0) {
                filePreviewArea.style.display = 'none';
            }
        };

        // If file is image, show thumbnail
        if (file.type.startsWith('image/')) {
            reader.onload = (e) => {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.alt = file.name;
                img.title = file.name;
                filePreviewWrapper.appendChild(img);
            };
            reader.readAsDataURL(file);
        } else {
            // For non-image files, show file type icon/placeholder
            const placeholder = document.createElement('div');
            const fileExt = file.name.split('.').pop().toLowerCase();
            const shortName = file.name.length > 15 ? file.name.substring(0, 12) + '...' : file.name;
            
            placeholder.textContent = shortName;
            placeholder.title = file.name;
            placeholder.classList.add('file-placeholder');
            
            // Add file type icon if known type
            if (['pdf', 'doc', 'docx', 'txt', 'xls', 'xlsx', 'ppt', 'pptx', 'zip'].includes(fileExt)) {
                placeholder.classList.add(`file-${fileExt}`);
            }
            
            filePreviewWrapper.appendChild(placeholder);
        }
        
        // Add remove button to preview
        filePreviewWrapper.appendChild(removeButton);
        
        // Show preview area and add the file preview
        filePreviewArea.style.display = 'flex';
        filePreviewArea.appendChild(filePreviewWrapper);
    }

    // Top bar actions
    if (btnNew) {
        btnNew.addEventListener('click', () => {
            fetch('/reset', { method: 'POST' })
                .then(r => r.json())
                .then(() => {
                    clearChat();
                    boot();
                    if (historyPanel) historyPanel.classList.remove('active');
                })
                .catch(() => {
                    clearChat();
                    boot();
                });
        });
    }

    // Toggle history panel
    if (btnHistory) {
        btnHistory.addEventListener('click', toggleHistoryPanel);
    }
    
    // Toggle maps panel
    if (btnMaps) {
        btnMaps.addEventListener('click', toggleMapsPanel);
    }
    
    // Close maps panel
    if (mapsCloseBtn) {
        mapsCloseBtn.addEventListener('click', () => {
            if (mapsPanel) mapsPanel.classList.remove('active');
        });
    }
    
    // Funkcje dla panelu wyszukiwania wiedzy
    function toggleSearchPanel() {
        if (!searchPanel) return;
        
        if (searchPanel.classList.contains('active')) {
            searchPanel.classList.remove('active');
        } else {
            searchPanel.classList.add('active');
            // Ukryj inne panele jeśli są otwarte
            if (historyPanel && historyPanel.classList.contains('active')) {
                historyPanel.classList.remove('active');
            }
            if (mapsPanel && mapsPanel.classList.contains('active')) {
                mapsPanel.classList.remove('active');
            }
            
            searchQueryInput.focus();
        }
    }
    
    // Wykonanie wyszukiwania
    function performSearch() {
        const query = searchQueryInput.value.trim();
        if (!query) return;
        
        // Pokaż ładowanie
        searchResults.innerHTML = '<div class="search-loading">Wyszukiwanie wiedzy...</div>';
        
        // Wykonaj zapytanie do API
        fetch(`/search?q=${encodeURIComponent(query)}`)
            .then(r => r.json())
            .then(data => {
                // Wyczyść ładowanie
                searchResults.innerHTML = '';
                
                if (data.error) {
                    searchResults.innerHTML = `<div class="search-error">Błąd wyszukiwania: ${data.error}</div>`;
                    return;
                }
                
                if (!data.results || data.results.length === 0) {
                    searchResults.innerHTML = '<div class="search-no-results">Nie znaleziono wyników</div>';
                    return;
                }
                
                // Wyświetl wyniki
                data.results.forEach(result => {
                    const resultElement = document.createElement('div');
                    resultElement.classList.add('search-result');
                    
                    resultElement.innerHTML = `
                        <div class="search-result-title">${result.title}</div>
                        <div class="search-result-url">${result.link}</div>
                        <div class="search-result-snippet">${result.snippet}</div>
                    `;
                    
                    searchResults.appendChild(resultElement);
                });
                
                // Dodaj komunikat do czatu
                addMessage(`Znalazłem ${data.results.length} wyników dla zapytania: "${query}"`, 'system');
                scrollToBottom();
            })
            .catch(error => {
                searchResults.innerHTML = `<div class="search-error">Błąd połączenia: ${error}</div>`;
            });
    }
    
    // Toggle search panel
    if (btnSearch) {
        btnSearch.addEventListener('click', toggleSearchPanel);
    }
    
    // Close search panel
    if (searchCloseBtn) {
        searchCloseBtn.addEventListener('click', () => {
            if (searchPanel) searchPanel.classList.remove('active');
        });
    }
    
    // Search button click
    if (searchBtn) {
        searchBtn.addEventListener('click', performSearch);
    }
    
    // Enter key in search input
    if (searchQueryInput) {
        searchQueryInput.addEventListener('keydown', e => {
            if (e.key === 'Enter') {
                e.preventDefault();
                performSearch();
            }
        });
    }
    
    // Use current location
    if (mapsGpsBtn) {
        mapsGpsBtn.addEventListener('click', useCurrentLocation);
    }
    
    // Find route
    if (mapsRouteBtn) {
        mapsRouteBtn.addEventListener('click', findRoute);
    }

    // Zamiana punktów A/B
    if (mapsSwapBtn) {
        mapsSwapBtn.addEventListener('click', () => {
            const a = mapsStartInput.value;
            const b = mapsEndInput.value;
            mapsStartInput.value = b;
            mapsEndInput.value = a;
        });
    }
    
    // Handle click outside panels to close them
    document.addEventListener('click', (event) => {
        if (historyPanel && historyPanel.classList.contains('active') && 
            !historyPanel.contains(event.target) && 
            event.target !== btnHistory) {
            historyPanel.classList.remove('active');
        }
        
        // Maps panel shouldn't close when clicking outside since it's a modal
    });
    
    // Initialize the chat interface
    function initChat() {
        // Connect to WebSocket
        connectWebSocket();
        
        // Show boot message with system info
        boot();
        
        // Set initial focus to message input
        if (messageInput) {
            setTimeout(() => messageInput.focus(), 500);
        }
        
        // Hide file preview area initially
        if (filePreviewArea) {
            filePreviewArea.style.display = 'none';
        }
    }
    
    // Start the chat
    initChat();
});
