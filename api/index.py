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
    # Endpoint unificado para todas las consultas del usuario
    url = "https://simongpa11.app.n8n.cloud/webhook-test/stop-reparto"
    async with httpx.AsyncClient() as client:
        try:
            print(f"DEBUG: Enviando mensaje de usuario a n8n: {query} (FlowStep: {flow_step})")
            # Payload limpio: texto actual, source: user y flow_step si existe
            payload = {
                "query": query,
                "source": "user"
            }
            if flow_step:
                payload["flow_step"] = flow_step

            response = await client.post(url, json=payload, timeout=8.0)
            if response.status_code == 200:
                return response.json()
            return {"status": "no_data"}
        except Exception as e:
            print(f"DEBUG: Error n8n: {str(e)}")
            return {"status": "error"}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        # Inicialización de recursos
        pc_key = clean_key(os.getenv('PINECONE_API_KEY'))
        index_name = clean_key(os.getenv('PINECONE_INDEX_NAME'))
        pc_namespace = clean_key(os.getenv('PINECONE_NAMESPACE', 'default'))
        google_key = clean_key(os.getenv('GOOGLE_API_KEY'))

        pc = Pinecone(api_key=pc_key)
        index = pc.Index(index_name)
        genai.configure(api_key=google_key)
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        
        # --- 1. LLAMADA CRÍTICA A N8N (CADA MENSAJE) ---
        n8n_data = await call_n8n_webhook(req.message, req.flow_step)
        
        # --- 2. BÚSQUEDA RAG (SIEMPRE ACTIVA) ---
        search_query = req.message
        if req.history:
            history_summary = "\n".join([f"{m.role}: {m.content}" for m in req.history[-2:]])
            rewrite_prompt = f"Basado en: {history_summary}\nReescribe '{req.message}' para búsqueda técnica. SOLO la frase."
            search_query = model.generate_content(rewrite_prompt).text.strip()

        embed = genai.embed_content(model='models/text-embedding-004', content=search_query, task_type="retrieval_query")
        results = index.query(vector=embed['embedding'], top_k=5, include_metadata=True, namespace=pc_namespace)
        
        context_parts = []
        sources = []
        seen = set()
        for m in results.matches:
            if m.score > 0.4:
                t = m.metadata.get('text', '')
                f = m.metadata.get('file_name', 'Manual')
                if t:
                    context_parts.append(t)
                    if f not in seen:
                        sources.append({"name": f, "score": round(m.score * 100, 1)})
                        seen.add(f)
        
        rag_context = "\n\n".join(context_parts)

        # --- 3. RESPUESTA FINAL CON COMBINACIÓN DE DATOS ---
        gemini_history = []
        for m in req.history:
            gemini_history.append({"role": "user" if m.role == "user" else "model", "parts": [m.content]})
        
        chat_session = model.start_chat(history=gemini_history)
        
        system_instr = f"""Eres un asistente experto de Aquaservice. 
        Tienes dos fuentes de verdad:
        
        1. DATOS EN TIEMPO REAL (n8n): {json.dumps(n8n_data)}
        2. MANUALES TÉCNICOS (RAG): {rag_context}
        
        INSTRUCCIONES:
        - Si n8n devuelve una confirmación de acción (como un stop de reparto), dalo por hecho y confírmalo al usuario cordialmente.
        - Si el usuario pregunta algo técnico, usa los manuales.
        - Sé breve, profesional y directo."""
        
        response = chat_session.send_message(f"{system_instr}\n\nUSUARIO: {req.message}")
        
        return {
            "answer": response.text,
            "rag": len(context_parts) > 0,
            "n8n": True,
            "sources": sources
        }
    except Exception as e:
        return {"answer": f"❌ Error en el sistema: {str(e)}", "rag": False, "sources": []}

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
        <title>Aquaservice AI v4.7 (Flow State n8n)</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Outfit', sans-serif; height: 100dvh; overscroll-behavior: none; }
            .dot { animation: pulse 1.4s infinite; }
            @keyframes pulse { 0%, 100% { opacity: .2; } 50% { opacity: 1; } }
        </style>
    </head>
    <body class="bg-gray-100 flex flex-col items-center">
        <div class="bg-white w-full md:max-w-2xl flex flex-col h-full md:h-[90vh] md:mt-8 md:rounded-3xl shadow-2xl overflow-hidden relative">
            <div class="bg-[#002E7D] p-4 md:p-6 text-white flex justify-between items-center shrink-0">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center text-xl">🚀</div>
                    <div>
                        <h1 class="text-base md:text-xl font-bold leading-tight">Aquaservice AI</h1>
                        <p class="text-blue-200 text-[10px] uppercase font-semibold">v4.7 Flow-State Logic</p>
                    </div>
                </div>
            </div>
            <div id="log" class="flex-1 overflow-y-auto p-4 md:p-6 space-y-4 bg-gray-50/50"></div>
            <div id="typing" class="hidden px-6 py-2 flex gap-2 items-center text-[#002E7D] text-xs font-medium">
                <span class="bg-blue-50 px-3 py-1 rounded-full border border-blue-100 italic font-semibold">Procesando...</span>
            </div>
            <div class="p-4 md:p-6 bg-white border-t border-gray-100 flex gap-2">
                <input id="q" type="text" enterkeyhint="send" autocomplete="off" class="flex-1 bg-gray-50 border border-gray-200 rounded-2xl px-4 py-3 outline-none focus:ring-2 focus:ring-[#002E7D] text-sm" placeholder="Escribe tu mensaje...">
                <button id="b" class="bg-[#002E7D] text-white p-3 md:p-4 rounded-2xl shadow-lg active:scale-95 transition-transform shrink-0">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M14 5l7 7-7 7M5 12h16" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
                </button>
            </div>
        </div>
        <script>
            let history = [];
            let flowStep = null; // Estado del flujo
            const log = document.getElementById('log');
            const q = document.getElementById('q');
            const b = document.getElementById('b');
            const typing = document.getElementById('typing');

            function addMsg(text, isUser, src = []) {
                const div = document.createElement('div');
                div.className = isUser ? "flex flex-row-reverse gap-3" : "flex gap-3";
                div.innerHTML = `
                    <div class="w-8 h-8 rounded-full flex items-center justify-center font-bold text-[10px] shrink-0 border ${isUser?'bg-[#002E7D] text-white':'bg-white text-[#002E7D]'}">${isUser?'U':'AI'}</div>
                    <div class="${isUser?'bg-[#002E7D] text-white rounded-tr-none':'bg-white border text-gray-700 rounded-tl-none'} p-4 rounded-2xl shadow-sm text-sm leading-relaxed max-w-[85%]">
                        ${text.replace(/\\n/g, '<br>')}
                        ${!isUser && src.length ? `<div class="mt-2 text-[10px] text-blue-400 border-t pt-2 uppercase font-bold">Manuales: ${src.map(s=>s.name).join(', ')}</div>` : ''}
                    </div>
                `;
                log.appendChild(div);
                log.scrollTop = log.scrollHeight;
                history.push({role: isUser ? "user" : "assistant", content: text});
            }

            async function ask() {
                const val = q.value.trim(); if(!val) return;
                
                // Preparamos el payload incluyendo el flowStep actual
                const payload = {
                    message: val, 
                    history: history.slice(-6),
                    flow_step: flowStep
                };

                addMsg(val, true); 
                q.value = '';
                typing.classList.remove('hidden');

                try {
                    const res = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(payload)
                    });
                    const d = await res.json();
                    typing.classList.add('hidden');
                    
                    // REGLA DE NEGOCIO: Si el bot pregunta por el motivo, activamos el estado
                    if (d.answer.includes("¿Cuál es el motivo del stop?")) {
                        flowStep = "awaiting_stop_reason";
                        console.log("DEBUG: FlowStep activado -> awaiting_stop_reason");
                    } else {
                        // Limpiamos el estado después de una interacción que no sea la pregunta
                        flowStep = null;
                    }

                    addMsg(d.answer, false, d.sources);
                } catch(e) {
                    typing.classList.add('hidden');
                    addMsg("Error de conexión.", false);
                }
            }
            b.onclick = ask;
            q.onkeypress = (e) => { if(e.key === 'Enter') ask(); };
            window.onload = () => addMsg("¡Hola! Estoy listo para ayudarte con tus manuales o gestiones de reparto.", false);
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
