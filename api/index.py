from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
import re
import httpx
import json
from typing import List, Optional

load_dotenv()

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []
    flow_step: Optional[str] = None

def clean_key(val):
    if not val: return ""
    if '=' in val: val = val.split('=')[-1]
    return re.sub(r'[\s\n\r\t]', '', val).strip("'\" ")

async def call_n8n_webhook(query: str, flow_step: Optional[str] = None):
    """
    Regla 1 y 3: Cada mensaje del usuario va al webhook con query y source: user.
    Regla 4: Se incluye flow_step si existe.
    """
    url = "https://simongpa11.app.n8n.cloud/webhook/salesforce-users"
    async with httpx.AsyncClient() as client:
        try:
            payload = {
                "query": query,
                "source": "user"
            }
            if flow_step:
                payload["flow_step"] = flow_step
            
            print(f"DEBUG: Enviando a n8n -> {json.dumps(payload)}")
            response = await client.post(url, json=payload, timeout=8.0)
            
            if response.status_code == 200:
                data = response.json()
                print(f"DEBUG: n8n respondió -> {json.dumps(data)}")
                return data
            return {"status": "no_response"}
        except Exception as e:
            print(f"DEBUG: Error n8n -> {str(e)}")
            return {"status": "error"}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        # Inicialización
        google_key = clean_key(os.getenv('GOOGLE_API_KEY'))
        pc_key = clean_key(os.getenv('PINECONE_API_KEY'))
        index_name = clean_key(os.getenv('PINECONE_INDEX_NAME'))
        pc_namespace = clean_key(os.getenv('PINECONE_NAMESPACE', 'default'))

        genai.configure(api_key=google_key)
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        
        # --- 1. LLAMADA A N8N (OBLIGATORIA POR CADA MENSAJE DE USUARIO) ---
        # Regla 1, 2 y 3: Solo el mensaje exacto del usuario.
        n8n_data = await call_n8n_webhook(req.message, req.flow_step)
        
        # --- 2. BÚSQUEDA RAG (CONTEXTO ADICIONAL) ---
        pc = Pinecone(api_key=pc_key)
        index = pc.Index(index_name)
        embed = genai.embed_content(model='models/text-embedding-004', content=req.message, task_type="retrieval_query")
        results = index.query(vector=embed['embedding'], top_k=3, include_metadata=True, namespace=pc_namespace)
        rag_context = "\n".join([m.metadata.get('text', '') for m in results.matches if m.score > 0.45])

        # --- 3. LÓGICA DE RESPUESTA ---
        # El backend (n8n) decide la lógica, Gemini la comunica cordialmente.
        system_instr = f"""Eres el asistente de Aquaservice. 
        REGLAS DE RESPUESTA:
        1. Tu lógica principal viene de n8n: {json.dumps(n8n_data)}
        2. Tu conocimiento técnico viene de RAG: {rag_context}
        
        Si n8n indica que se requiere el motivo, pregunta exactamente: '¿Cuál es el motivo del stop?'
        Si n8n ya confirma una acción, comunícala al usuario.
        Si no hay información en n8n ni RAG, responde amablemente que no puedes ayudar con eso todavía."""
        
        # Regla 2: El historial se usa para el modelo, pero NUNCA se envía el historial a n8n.
        gemini_history = [{"role": "user" if m.role == "user" else "model", "parts": [m.content]} for m in req.history]
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(f"{system_instr}\n\nUsuario dice: {req.message}")
        
        # --- 4. GESTIÓN DE ESTADO (FLOW_STEP) ---
        new_flow_step = None
        
        # Normalizamos n8n_data si viene como lista (común en n8n)
        resolved_n8n = n8n_data[0] if isinstance(n8n_data, list) and len(n8n_data) > 0 else n8n_data
        
        # Si n8n nos manda un flag o si Gemini genera la pregunta clave
        if "¿Cuál es el motivo del stop?" in response.text:
            new_flow_step = "awaiting_stop_reason"
        elif isinstance(resolved_n8n, dict) and resolved_n8n.get("next_step") == "awaiting_stop_reason":
            new_flow_step = "awaiting_stop_reason"

        return {
            "answer": response.text,
            "flow_step": new_flow_step,
            "sources": [{"name": m.metadata.get('file_name', 'Manual'), "score": round(m.score*100,1)} for m in results.matches if m.score > 0.45]
        }
    except Exception as e:
        return {"answer": f"❌ Error: {str(e)}", "flow_step": None}

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
        <title>Aquaservice AI v4.8 (Stateful n8n)</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Outfit', sans-serif; height: 100dvh; overscroll-behavior: none; }
        </style>
    </head>
    <body class="bg-gray-100 flex flex-col items-center">
        <div class="bg-white w-full md:max-w-2xl flex flex-col h-full md:h-[90vh] md:mt-8 md:rounded-3xl shadow-2xl overflow-hidden relative">
            
            <div class="bg-[#002E7D] p-4 text-white flex justify-between items-center shrink-0">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center text-xl">🤖</div>
                    <div>
                        <h1 class="text-base font-bold leading-tight">Gestor de Reparto</h1>
                        <p class="text-blue-200 text-[10px] uppercase font-semibold">Integración n8n Activa</p>
                    </div>
                </div>
            </div>

            <div id="log" class="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50/50"></div>

            <div id="typing" class="hidden px-6 py-2 text-[#002E7D] text-xs font-semibold">
                <span class="animate-pulse">Consultando sistemas...</span>
            </div>

            <div class="p-4 bg-white border-t flex gap-2">
                <input id="q" type="text" autocomplete="off" class="flex-1 bg-gray-50 border rounded-2xl px-4 py-3 text-sm outline-none" placeholder="Escribe tu mensaje...">
                <button id="b" class="bg-[#002E7D] text-white p-3 rounded-2xl shadow-lg">➤</button>
            </div>
        </div>

        <script>
            let history = [];
            let currentFlowStep = null; // ESTADO LOCAL flow_step

            function addMsg(text, isUser, sources = []) {
                const div = document.createElement('div');
                div.className = isUser ? "flex flex-row-reverse gap-3" : "flex gap-3";
                
                let sHtml = "";
                if (!isUser && sources && sources.length > 0) {
                    sHtml = `
                        <div class="mt-3 pt-2 border-t border-gray-100 flex justify-end">
                            <div class="relative has-tooltip group">
                                <div class="bg-blue-50 text-blue-600 rounded-full w-5 h-5 flex items-center justify-center text-[10px] cursor-help font-bold border border-blue-100">i</div>
                                <div class="tooltip invisible group-hover:visible opacity-0 group-hover:opacity-100 transition-opacity bg-slate-800 text-white p-3 rounded-xl shadow-2xl absolute bottom-full right-0 mb-2 w-48 z-50 text-[10px]">
                                    <p class="font-bold border-b border-white/10 mb-2 pb-1 text-blue-300 uppercase">Fuentes de conocimiento</p>
                                    ${sources.map(s => `<div class="flex justify-between mb-1"><span>${s.name}</span><span class="text-blue-300 ml-2">${s.score}%</span></div>`).join('')}
                                </div>
                            </div>
                        </div>
                    `;
                }

                div.innerHTML = `
                    <div class="w-8 h-8 rounded-full flex items-center justify-center text-[10px] shrink-0 ${isUser?'bg-[#002E7D] text-white':'bg-white border text-[#002E7D]'} font-bold">${isUser?'U':'AI'}</div>
                    <div class="${isUser?'bg-[#002E7D] text-white rounded-tr-none':'bg-white border text-gray-700 rounded-tl-none'} p-4 rounded-2xl text-sm max-w-[85%] shadow-sm">
                        ${text.replace(/\\n/g, '<br>')}
                        ${sHtml}
                    </div>
                `;
                document.getElementById('log').appendChild(div);
                document.getElementById('log').scrollTop = document.getElementById('log').scrollHeight;
                history.push({role: isUser ? "user" : "assistant", content: text});
            }

            async function ask() {
                const q = document.getElementById('q');
                const val = q.value.trim(); if(!val) return;
                
                addMsg(val, true);
                q.value = '';
                document.getElementById('typing').classList.remove('hidden');

                try {
                    const res = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            message: val, 
                            history: history.slice(-6),
                            flow_step: currentFlowStep
                        })
                    });
                    const d = await res.json();
                    
                    currentFlowStep = d.flow_step;
                    document.getElementById('typing').classList.add('hidden');
                    addMsg(d.answer, false, d.sources);
                } catch(e) {
                    document.getElementById('typing').classList.add('hidden');
                    addMsg("Error técnico.", false);
                }
            }

            document.getElementById('b').onclick = ask;
            document.getElementById('q').onkeypress = (e) => { if(e.key === 'Enter') ask(); };
            window.onload = () => addMsg("¡Hola! Soy tu gestor de Aquaservice. ¿En qué puedo ayudarte hoy?", false);
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
