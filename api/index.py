from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import os
import json
from typing import List, Optional, Any, Dict
from dotenv import load_dotenv

# Import our new Agent System
from api.agent_system import AgentSystem

load_dotenv()

app = FastAPI()

# Global Agent System Instance
agent_system = None

def get_agent_system():
    global agent_system
    if not agent_system:
        agent_system = AgentSystem()
    return agent_system

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []
    flow_step: Optional[str] = None
    context: Optional[Dict[str, Any]] = {}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        sys = get_agent_system()
        print(f"DEBUG: Processing with Context: {req.context}")
        
        response_data = await sys.process_request(
            message=req.message,
            history=[m.dict() for m in req.history],
            context=req.context,
            flow_step=req.flow_step
        )
        
        return {
            "answer": response_data.get("answer", "No answer generated."),
            "flow_step": response_data.get("flow_step"),
            "sources": response_data.get("sources_data", []), 
            "context_updates": response_data.get("context_updates", {}),
            "source_type": response_data.get("source"),
            "debug_info": response_data.get("debug_info")
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"answer": f"❌ Error del Sistema: {str(e)}", "flow_step": None}

@app.post("/api/tts")
async def text_to_speech(req: Message):
    try:
        import httpx
        text = req.content
        voice_id = "7QQzpAyzlKTVrRzQJmTE" # Custom Voice
        api_key = os.getenv("ELEVENLABS_API_KEY")
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=data, headers=headers, timeout=30.0)
            if resp.status_code != 200:
                print(f"ElevenLabs Error: {resp.text}")
                return JSONResponse(status_code=500, content={"message": "TTS Error"})
            from fastapi.responses import Response
            return Response(content=resp.content, media_type="audio/mpeg")

    except Exception as e:
        print(f"TTS Exception: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
        <title>Aquaservice Agentic System</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <script src="https://unpkg.com/lucide@latest"></script>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            body { font-family: 'Outfit', sans-serif; height: 100dvh; overscroll-behavior: none; }
            .scrollbar-hide::-webkit-scrollbar { display: none; }
            .glass { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); }
            
            .toggle-checkbox:checked { right: 0; border-color: #002E7D; }
            .toggle-checkbox:checked + .toggle-label { background-color: #002E7D; }
            
            .prose p { margin-bottom: 0.5em; }
            .prose strong { color: inherit; font-weight: 700; }
            .prose ul { list-style-type: disc; padding-left: 1.2em; margin-bottom: 0.5em; }
            .prose ol { list-style-type: decimal; padding-left: 1.2em; margin-bottom: 0.5em; }

            /* Voice Pulse Animation */
            .mic-wrapper {
                position: relative;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .mic-button {
                z-index: 20;
                transition: all 0.3s ease;
            }
            .pulse-ring {
                position: absolute;
                top: 50%; left: 50%;
                transform: translate(-50%, -50%);
                width: 100%; height: 100%;
                border-radius: 50%;
                background: rgba(0, 46, 125, 0.2);
                z-index: 10;
                animation: ripple 1.5s infinite cubic-bezier(0.4, 0, 0.6, 1);
            }
            @keyframes ripple {
                0% { transform: translate(-50%, -50%) scale(0.8); opacity: 1; }
                100% { transform: translate(-50%, -50%) scale(1.6); opacity: 0; }
            }
            
            .speaking-wave { display: flex; align-items: center; gap: 4px; height: 20px; }
            .speaking-wave span { display: block; width: 4px; background: #002E7D; animation: wave 1s infinite ease-in-out; }
            .speaking-wave span:nth-child(2) { animation-delay: 0.1s; }
            .speaking-wave span:nth-child(3) { animation-delay: 0.2s; }
            .speaking-wave span:nth-child(4) { animation-delay: 0.3s; }
            @keyframes wave { 0%, 100% { height: 4px; } 50% { height: 20px; } }
        </style>
    </head>
    <body class="bg-slate-100 flex items-center justify-center p-0 md:p-4">
        
        <div class="bg-white w-full h-full md:max-w-6xl md:h-[90vh] md:rounded-3xl shadow-2xl overflow-hidden flex relative">
            
            <!-- Context Panel -->
            <div id="settingsPanel" class="hidden md:flex flex-col w-72 bg-slate-50 border-r border-slate-200 h-full transition-all duration-300 absolute md:relative z-20 md:z-0 transform md:transform-none -translate-x-full md:translate-x-0">
                <!-- ... settings content ... -->
                <div class="p-6 border-b border-slate-200 flex justify-between items-center bg-white">
                    <h2 class="font-bold text-[#002E7D] text-lg">Contexto Cliente</h2>
                    <button id="closeSettings" class="md:hidden text-slate-400 hover:text-red-500"><i data-lucide="x"></i></button>
                </div>
                <div class="flex-1 overflow-y-auto p-6 space-y-6">
                    <form id="contextForm" class="space-y-4"></form>
                </div>
                <div class="p-4 border-t border-slate-200 bg-white">
                    <button onclick="resetContext()" class="w-full py-2 text-slate-500 text-sm hover:text-[#002E7D] font-medium transition-colors">Restablecer Valores</button>
                </div>
            </div>

            <!-- Main Content -->
            <div class="flex-1 flex flex-col h-full relative w-full border-r border-slate-100 bg-white">
                
                <!-- Navbar -->
                <!-- ... navbar code ... -->
                <div class="bg-white/80 backdrop-blur-md px-4 pt-4 border-b border-slate-100 flex justify-between items-center shrink-0 w-full z-10">
                    <div class="flex items-center gap-4">
                         <button id="toggleSettings" class="md:hidden p-2 rounded-xl bg-slate-100 text-[#002E7D]"><i data-lucide="settings-2"></i></button>
                         <div class="flex bg-slate-100 p-1 rounded-2xl">
                             <button onclick="switchTab('chat')" id="tab-chat" class="px-4 py-2 rounded-xl text-sm font-bold transition-all bg-white text-[#002E7D] shadow-sm">Chat</button>
                             <button onclick="switchTab('voice')" id="tab-voice" class="px-4 py-2 rounded-xl text-sm font-bold transition-all text-slate-500 hover:text-[#002E7D]">Voz (Teléfono)</button>
                         </div>
                    </div>
                    <div class="flex gap-2 mb-2">
                        <button onclick="toggleDebug()" class="p-2 text-slate-400 hover:text-[#002E7D] transition-colors" title="Toggle Logic View"><i data-lucide="cpu" size="18"></i></button>
                        <button onclick="location.reload()" class="p-2 text-slate-400 hover:text-[#002E7D] transition-colors"><i data-lucide="rotate-ccw" size="18"></i></button>
                    </div>
                </div>

                <!-- Chat View -->
                <div id="view-chat" class="flex-1 flex flex-col h-full overflow-hidden">
                    <div id="log" class="flex-1 overflow-y-auto p-4 space-y-6 bg-slate-50 scroll-smooth">
                        <div class="flex gap-4">
                             <div class="w-8 h-8 rounded-full bg-gradient-to-br from-[#002E7D] to-blue-600 flex items-center justify-center text-white text-xs font-bold shrink-0 mt-1">AI</div>
                             <div class="bg-white border border-slate-100 p-4 rounded-2xl rounded-tl-none text-slate-600 text-[15px] shadow-sm max-w-[85%] leading-relaxed prose">
                                Hola 👋 Soy tu asistente inteligente.
                             </div>
                        </div>
                    </div>
                    <div id="typing" class="hidden px-6 py-2 bg-slate-50">
                        <div class="flex items-center gap-2 text-blue-600 text-xs font-semibold bg-blue-50 w-fit px-3 py-1.5 rounded-full">
                            <i data-lucide="loader-2" class="animate-spin" size="12"></i><span>Procesando...</span>
                        </div>
                    </div>
                    <div class="p-4 bg-white/80 backdrop-blur-md border-t border-slate-100">
                        <div class="flex gap-2 max-w-3xl mx-auto relative">
                            <input id="q" type="text" autocomplete="off" class="flex-1 bg-slate-100 border-0 rounded-2xl px-5 py-4 text-slate-700 outline-none text-[15px]" placeholder="Escribe tu mensaje aquí...">
                            <button id="b" class="bg-[#002E7D] hover:bg-blue-900 text-white px-6 rounded-2xl shadow-lg transition-all flex items-center justify-center"><i data-lucide="send" size="20"></i></button>
                        </div>
                        <div class="text-center mt-2"><span id="activeSource" class="text-[10px] uppercase font-bold text-slate-300 tracking-wider"></span></div>
                    </div>
                </div>

                <!-- Voice View -->
                <div id="view-voice" class="hidden flex-1 flex flex-col items-center justify-center bg-slate-50 relative overflow-hidden">
                    <div class="absolute inset-0 opacity-10 pointer-events-none flex items-center justify-center">
                        <div class="w-96 h-96 bg-[#002E7D] rounded-full blur-3xl"></div>
                    </div>
                    <div class="z-10 text-center space-y-8 flex flex-col items-center">
                        <div class="mb-4">
                            <h2 class="text-2xl font-bold text-slate-800">Llamada en curso</h2>
                            <p class="text-slate-500" id="voiceStatus">Presiona para hablar</p>
                        </div>
                        <div class="mic-wrapper w-32 h-32">
                            <button id="micBtn" onclick="toggleVoice()" class="mic-button w-24 h-24 rounded-full bg-white shadow-2xl flex items-center justify-center border-4 border-white">
                                <i id="micIcon" data-lucide="mic" class="text-slate-400 w-10 h-10"></i>
                            </button>
                            <div id="voiceRing" class="hidden"></div>
                        </div>
                        <div id="wave-container" class="h-8 flex justify-center items-center opacity-0 transition-opacity">
                            <div class="speaking-wave"><span></span><span></span><span></span><span></span></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Debug Panel -->
            <!-- ... -->
            <div id="debugPanel" class="hidden md:flex flex-col w-80 bg-slate-900 border-l border-slate-800 h-full overflow-y-auto text-slate-300 font-mono text-xs">
                 <div class="p-4 border-b border-slate-700 bg-slate-950 flex justify-between items-center sticky top-0">
                    <h2 class="font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-2"><i data-lucide="terminal" size="14"></i> Logic Trace</h2>
                </div>
                <div id="debugLog" class="p-4 space-y-4">
                    <div class="text-slate-500 italic text-center mt-10">Esperando eventos...</div>
                </div>
            </div>
        </div>

        <div id="overlay" class="fixed inset-0 bg-black/50 z-10 hidden md:hidden glass"></div>

        <script>
            lucide.createIcons();
            
            // --- STATE ---
            const defaultContext = {
              "plan": "Ahorro",
              "scoring": 3.5,
              "motivo": "exceso_agua",
              "stops_ultimo_ano": 1,
              "albaran_descargado": false,
              "tipo_cliente": "residencial",
              "canal": "Chat",
              "is_delivery_day": false,
              "has_pending_usual_delivery": false,
              "has_pending_crm_delivery": false,
              "urgent_notice_allowed_zone": true,
              "next_delivery_hours": 72,
              "route_type": "Normal",
              "pending_crm_hours": 24,
              "es_puntual": false,
              "producto": "agua",
              "cantidad": ""
            };
            let userContext = {...defaultContext};
            let flowStep = null;
            let historyLog = [];
            let showDebug = true;
            let currentTab = 'chat';
            let isRecording = false;
            let recognition;
            let currentAudio = null;
            let autoListen = false;

            // --- UI FUNCTIONS ---
            function renderSettings() {
                const form = document.getElementById('contextForm');
                if (!form) return;
                form.innerHTML = '';
                Object.keys(userContext).forEach(key => {
                    const val = userContext[key];
                    const type = typeof val;
                    const group = document.createElement('div');
                    group.className = "flex flex-col gap-1.5";
                    const label = document.createElement('label');
                    label.className = "text-xs font-bold text-slate-500 uppercase tracking-wide";
                    label.innerText = key.replace(/_/g, ' ');
                    
                    let input;
                    if (type === 'boolean') {
                        const wrapper = document.createElement('div');
                        wrapper.className = "relative inline-block w-10 mr-2 align-middle select-none transition duration-200 ease-in";
                        input = document.createElement('input');
                        input.type = "checkbox"; input.checked = val;
                        input.className = "toggle-checkbox absolute block w-5 h-5 rounded-full bg-white border-2 border-slate-300 appearance-none cursor-pointer transition-all duration-300";
                        const lbl = document.createElement('label');
                        lbl.className = "toggle-label block overflow-hidden h-5 rounded-full bg-slate-300 cursor-pointer";
                        wrapper.appendChild(input); wrapper.appendChild(lbl);
                        const row = document.createElement('div');
                        row.className = "flex items-center justify-between bg-white p-2 border rounded-xl";
                        row.appendChild(label); row.appendChild(wrapper);
                        input.onchange = (e) => updateContext(key, e.target.checked);
                        group.appendChild(row); form.appendChild(group); return;
                    } 
                    if (type === 'number') {
                        input = document.createElement('input'); input.type = "number"; input.value = val; input.step = "0.1";
                    } else {
                        if (key === 'plan') {
                             input = document.createElement('select');
                             ['Ahorro', 'Planocho'].forEach(opt => {
                                const o = document.createElement('option'); o.value = opt; o.text = opt; o.selected = opt === val; input.appendChild(o);
                             });
                        } else {
                            input = document.createElement('input'); input.type = "text"; input.value = val;
                        }
                    }
                    if (!input.className) {
                        input.className = "w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm text-slate-700 outline-none";
                        input.onchange = (e) => updateContext(key, type === 'number' ? parseFloat(e.target.value) : e.target.value);
                    }
                    group.appendChild(label); group.appendChild(input); form.appendChild(group);
                });
            }

            function updateContext(key, value) { userContext[key] = value; console.log("Ctx Upd:", key, value); }
            function resetContext() { userContext = {...defaultContext}; renderSettings(); }
            function syncContext(updates) {
                if (!updates) return;
                let changed = false;
                Object.keys(updates).forEach(k => { if (userContext[k] !== updates[k]) { userContext[k] = updates[k]; changed = true; }});
                if (changed) renderSettings();
            }

            function addMsg(text, isUser, meta = {}) {
                const log = document.getElementById('log');
                const div = document.createElement('div');
                div.className = isUser ? "flex flex-row-reverse gap-4" : "flex gap-4";
                let sourceBadge = '';
                if (!isUser && meta.source) {
                    let color = 'bg-slate-200 text-slate-600';
                    if(meta.source.includes('STOP')) color = 'bg-red-100 text-red-600';
                    if(meta.source.includes('URGENT')) color = 'bg-amber-100 text-amber-600';
                    if(meta.source.includes('RAG')) color = 'bg-emerald-100 text-emerald-600';
                    sourceBadge = `<span class="text-[10px] font-bold px-2 py-0.5 rounded-md ${color} ml-2 mb-1 inline-block uppercase tracking-wider">${meta.source.replace('AGENT_', '')}</span>`;
                }
                div.innerHTML = `
                    <div class="w-8 h-8 rounded-full flex items-center justify-center text-xs shrink-0 mt-1 shadow-sm ${isUser?'bg-slate-200 text-slate-600':'bg-gradient-to-br from-[#002E7D] to-blue-600 text-white font-bold'}">${isUser?'TÚ':'AI'}</div>
                    <div class="flex flex-col ${isUser ? 'items-end' : 'items-start'} max-w-[85%]">
                         ${sourceBadge}
                        <div class="${isUser?'bg-[#002E7D] text-white rounded-tr-none':'bg-white border border-slate-100 text-slate-600 rounded-tl-none'} p-4 rounded-2xl text-[15px] shadow-sm leading-relaxed prose">
                            ${isUser ? text.replace(/\\n/g, '<br>') : marked.parse(text)}
                        </div>
                    </div>`;
                log.appendChild(div); log.scrollTop = log.scrollHeight;
                historyLog.push({role: isUser ? "user" : "assistant", content: text});
            }

            async function ask() {
                const q = document.getElementById('q'); const val = q.value.trim(); if(!val) return;
                addMsg(val, true); q.value = '';
                document.getElementById('typing').classList.remove('hidden');
                try {
                    const res = await fetch('/api/chat', {
                        method: 'POST', headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ message: val, history: historyLog.slice(-6), flow_step: flowStep, context: userContext })
                    });
                    const d = await res.json();
                    flowStep = d.flow_step;
                    document.getElementById('typing').classList.add('hidden');
                    addMsg(d.answer, false, {source: d.source_type});
                    if (d.context_updates) syncContext(d.context_updates);
                    if (d.debug_info) addDebugTrace(d.debug_info);
                    const srcLabel = document.getElementById('activeSource');
                    if(srcLabel) srcLabel.innerText = "Último proceso: " + (d.source_type || "N/A");
                } catch(e) { console.error(e); document.getElementById('typing').classList.add('hidden'); addMsg("Error de conexión.", false); }
            }

            // --- VOICE LOGIC ---
            if ('webkitSpeechRecognition' in window) {
                recognition = new webkitSpeechRecognition();
                recognition.continuous = false; recognition.lang = 'es-ES'; recognition.interimResults = false;
                
                recognition.onstart = () => { isRecording = true; updateVoiceUI('listening'); };
                
                recognition.onresult = (e) => {
                    const transcript = e.results[0][0].transcript;
                    handleVoiceInput(transcript);
                };
                
                recognition.onerror = (e) => { 
                    console.error(e);
                    if (e.error === 'no-speech' && autoListen) {
                         // Keep alive retry logic
                         setTimeout(() => { if(autoListen && !isRecording) recognition.start(); }, 500);
                         return;
                    }
                    updateVoiceUI('idle'); isRecording = false; 
                };
                
                recognition.onend = () => { isRecording = false; };
            }

            function switchTab(tab) {
                currentTab = tab;
                document.getElementById('view-chat').classList.toggle('hidden', tab !== 'chat');
                document.getElementById('view-voice').classList.toggle('hidden', tab !== 'voice');
                const btnChat = document.getElementById('tab-chat');
                const btnVoice = document.getElementById('tab-voice');
                if (tab === 'chat') {
                    btnChat.className = "px-4 py-2 rounded-xl text-sm font-bold bg-white text-[#002E7D] shadow-sm transition-all";
                    btnVoice.className = "px-4 py-2 rounded-xl text-sm font-bold text-slate-500 hover:text-[#002E7D] transition-all";
                } else {
                    btnVoice.className = "px-4 py-2 rounded-xl text-sm font-bold bg-white text-[#002E7D] shadow-sm transition-all";
                    btnChat.className = "px-4 py-2 rounded-xl text-sm font-bold text-slate-500 hover:text-[#002E7D] transition-all";
                }
            }

            function toggleVoice() {
                if (!recognition) return alert("Navegador no soportado");
                if (autoListen || isRecording) {
                    autoListen = false;
                    recognition.stop();
                    if(currentAudio) { currentAudio.pause(); currentAudio = null; }
                    updateVoiceUI('idle');
                } else {
                    autoListen = true;
                    if(currentAudio) { currentAudio.pause(); currentAudio = null; } 
                    try { recognition.start(); } catch(e) { console.error(e); }
                }
            }

            function updateVoiceUI(state) {
                const ring = document.getElementById('voiceRing'); 
                const micIcon = document.getElementById('micIcon');
                const status = document.getElementById('voiceStatus'); 
                const wave = document.getElementById('wave-container');
                const micBtn = document.getElementById('micBtn');
                
                ring.className = "hidden";
                micBtn.className = "mic-button w-24 h-24 rounded-full bg-white shadow-2xl flex items-center justify-center border-4 border-white";
                micIcon.className = "w-10 h-10 text-slate-400";
                wave.classList.remove('opacity-100'); wave.classList.add('opacity-0');

                if (state === 'listening') {
                    status.innerText = "Escuchando..."; 
                    micBtn.className = "mic-button w-24 h-24 rounded-full bg-red-500 shadow-xl flex items-center justify-center border-4 border-red-100";
                    micIcon.className = "w-10 h-10 text-white";
                    ring.className = "pulse-ring"; 
                } else if (state === 'processing') {
                    status.innerText = "Pensando..."; 
                    micIcon.className = "w-10 h-10 text-slate-400 animate-pulse";
                } else if (state === 'speaking') {
                    status.innerText = "Pablo hablando..."; 
                    micBtn.className = "mic-button w-24 h-24 rounded-full bg-[#002E7D] shadow-xl flex items-center justify-center border-4 border-blue-100";
                    micIcon.className = "w-10 h-10 text-white";
                    wave.classList.remove('opacity-0'); wave.classList.add('opacity-100');
                } else {
                    status.innerText = "Presiona para hablar";
                }
                lucide.createIcons();
            }

            async function handleVoiceInput(text) {
                updateVoiceUI('processing');
                try {
                     const res = await fetch('/api/chat', {
                        method: 'POST', headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ message: text, history: historyLog.slice(-6), flow_step: flowStep, context: userContext })
                    });
                    const d = await res.json();
                    flowStep = d.flow_step;
                    if (d.context_updates) syncContext(d.context_updates);
                    if (d.debug_info) addDebugTrace(d.debug_info);
                    addDebugTrace({user_voice: text, assistant_text: d.answer});
                    historyLog.push({role: "user", content: text}); historyLog.push({role: "assistant", content: d.answer});

                    const ttsRes = await fetch('/api/tts', {
                        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ role: 'assistant', content: d.answer })
                    });
                    if (ttsRes.ok) {
                        const blob = await ttsRes.blob(); 
                        const url = URL.createObjectURL(blob);
                        currentAudio = new Audio(url);
                        currentAudio.onplay = () => updateVoiceUI('speaking');
                        currentAudio.onended = () => {
                            if (autoListen) {
                                updateVoiceUI('listening');
                                try { recognition.start(); } catch(e) { setTimeout(() => { if(autoListen) recognition.start(); }, 200); }
                            } else { updateVoiceUI('idle'); }
                        };
                        currentAudio.play();
                    } else { 
                        updateVoiceUI('idle'); autoListen = false;
                    }
                } catch (e) { updateVoiceUI('idle'); autoListen = false; }
            }

            function toggleDebug() {
                const p = document.getElementById('debugPanel'); p.classList.toggle('hidden'); p.classList.toggle('flex');
            }
            function addDebugTrace(data) {
                if (!data) return;
                const container = document.getElementById('debugLog');
                if (container.children.length === 1 && container.children[0].classList.contains('text-slate-500')) container.innerHTML = '';
                const entry = document.createElement('div');
                entry.className = "bg-slate-800/50 rounded-lg p-3 border border-slate-700/50 break-words mb-2";
                const time = new Date().toLocaleTimeString();
                let content = `<div class="text-[10px] text-slate-500 mb-1 flex justify-between"><span>EVENT</span><span>${time}</span></div>`;
                for (const [key, val] of Object.entries(data)) {
                    if (key === 'context_updates') continue;
                    let displayVal = val; if (typeof val === 'object') displayVal = JSON.stringify(val, null, 2);
                    content += `<div class="mb-1"><span class="text-slate-500 uppercase text-[10px] font-bold block">${key}</span><span class="text-slate-300 whitespace-pre-wrap">${displayVal}</span></div>`;
                }
                entry.innerHTML = content; container.insertBefore(entry, container.firstChild);
            }

            document.getElementById('b').onclick = ask;
            document.getElementById('q').onkeypress = (e) => { if(e.key === 'Enter') ask(); };
            const panel = document.getElementById('settingsPanel'); const overlay = document.getElementById('overlay');
            const toggleBtn = document.getElementById('toggleSettings'); const closeBtn = document.getElementById('closeSettings');
            function togglePanel() { panel.classList.toggle('-translate-x-full'); panel.classList.toggle('absolute'); overlay.classList.toggle('hidden'); }
            toggleBtn.onclick = togglePanel; closeBtn.onclick = togglePanel; overlay.onclick = togglePanel;

            renderSettings();
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
