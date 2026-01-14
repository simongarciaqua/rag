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

def clean_key(val):
    if not val: return ""
    if '=' in val: val = val.split('=')[-1]
    return re.sub(r'[\s\n\r\t]', '', val).strip("'\" ")

async def call_salesforce_n8n(query: str):
    url = "https://simongpa11.app.n8n.cloud/webhook/salesforce-users"
    async with httpx.AsyncClient() as client:
        try:
            # Enviamos la consulta al webhook de n8n
            response = await client.post(url, json={"query": query}, timeout=10.0)
            if response.status_code == 200:
                return response.json()
            return {"error": f"Error API: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        pc_key = clean_key(os.getenv('PINECONE_API_KEY'))
        index_name = clean_key(os.getenv('PINECONE_INDEX_NAME'))
        pc_namespace = clean_key(os.getenv('PINECONE_NAMESPACE', 'default'))
        google_key = clean_key(os.getenv('GOOGLE_API_KEY'))

        pc = Pinecone(api_key=pc_key)
        index = pc.Index(index_name)
        genai.configure(api_key=google_key)
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        
        # --- 1. DETECCIÓN DE INTENCIÓN ---
        # Decidimos si es RAG (Manuales) o SALESFORCE (Datos de usuario)
        history_summary = "\n".join([f"{m.role}: {m.content}" for m in req.history[-3:]])
        intent_prompt = f"""Historial: {history_summary}\nPregunta: {req.message}\n
        Analiza si el usuario pregunta por:
        A) Información general, técnica o de procesos (ej: como limpiar, que es Aquaservice, productos).
        B) Información personal, de su cuenta, sus facturas o datos de Salesforce (ej: mis datos, mi última factura, quien soy).
        Responde SOLO con la letra 'A' o 'B'."""
        
        intent_res = model.generate_content(intent_prompt).text.strip()
        is_salesforce = 'B' in intent_res

        context_data = ""
        sources = []
        source_type = "RAG"

        if is_salesforce:
            # --- RUTA B: SALESFORCE (n8n) ---
            sf_data = await call_salesforce_n8n(req.message)
            context_data = f"DATOS DE SALESFORCE (FACTURAS/USUARIO):\n{json.dumps(sf_data, indent=2)}"
            source_type = "Salesforce"
            sources = [{"name": "Salesforce API", "score": 100}]
        else:
            # --- RUTA A: RAG (Pinecone) ---
            search_query = req.message
            if req.history:
                rewrite_prompt = f"Convierte esta pregunta en una búsqueda para manuales: '{req.message}'. SOLO la búsqueda."
                search_query = model.generate_content(rewrite_prompt).text.strip()

            embed = genai.embed_content(model='models/text-embedding-004', content=search_query, task_type="retrieval_query")
            results = index.query(vector=embed['embedding'], top_k=5, include_metadata=True, namespace=pc_namespace)
            
            cp = []
            seen = set()
            for m in results.matches:
                if m.score > 0.4:
                    t = m.metadata.get('text', '')
                    f = m.metadata.get('file_name', 'Manual')
                    if t:
                        cp.append(t)
                        if f not in seen:
                            sources.append({"name": f, "score": round(m.score * 100, 1)})
                            seen.add(f)
            context_data = "\n\n".join(cp)

        # --- 3. RESPUESTA FINAL ---
        gemini_history = []
        for m in req.history:
            gemini_history.append({"role": "user" if m.role == "user" else "model", "parts": [m.content]})
        
        chat_session = model.start_chat(history=gemini_history)
        system_instr = f"""Eres un asistente de Aquaservice. 
        Hoy tienes acceso a esta información recuperada de {source_type}:
        {context_data}
        
        Usa estos datos para responder. Si no hay datos relevantes, intenta ayudar con lo que sepas pero indica que no encontraste información específica."""
        
        response = chat_session.send_message(f"{system_instr}\n\nPregunta: {req.message}")
        
        return {
            "answer": response.text,
            "rag": not is_salesforce,
            "salesforce": is_salesforce,
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
        <title>Aquaservice AI v4.0 (RAG + Salesforce)</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Outfit', sans-serif; height: 100dvh; overscroll-behavior: none; }
            .dot { animation: pulse 1.4s infinite; }
            @keyframes pulse { 0%, 100% { opacity: .2; } 50% { opacity: 1; } }
            .tooltip { visibility: hidden; opacity: 0; transition: opacity 0.2s; position: absolute; bottom: 120%; right: 0; width: 200px; }
            .has-tooltip:hover .tooltip { visibility: visible; opacity: 1; }
        </style>
    </head>
    <body class="bg-gray-100 flex flex-col items-center">
        <div class="bg-white w-full md:max-w-2xl flex flex-col h-full md:h-[90vh] md:mt-8 md:rounded-3xl shadow-2xl overflow-hidden relative">
            
            <div class="bg-[#002E7D] p-4 md:p-6 text-white flex justify-between items-center shrink-0 z-30">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center text-xl">🚀</div>
                    <div>
                        <h1 class="text-base md:text-xl font-bold leading-tight">Aquaservice AI</h1>
                        <p class="text-blue-200 text-[10px] uppercase font-semibold">v4.0 Híbrido (RAG + Salesforce)</p>
                    </div>
                </div>
                <div id="source-badge" class="text-[9px] px-2 py-0.5 rounded-full border border-white/30 bg-white/10 uppercase tracking-widest hidden"></div>
            </div>

            <div id="log" class="flex-1 overflow-y-auto p-4 md:p-6 space-y-4 bg-gray-50/50"></div>

            <div class="bg-white border-t border-gray-100 z-20 pb-safe">
                <div id="typing" class="hidden px-4 py-2">
                    <span class="bg-blue-50 text-[#002E7D] text-[10px] px-3 py-1 rounded-full border border-blue-100">Analizando consulta...</span>
                </div>
                <div class="p-4 md:p-6 flex gap-2 items-center">
                    <input id="q" type="text" enterkeyhint="send" autocomplete="off" class="flex-1 bg-gray-50 border border-gray-200 rounded-2xl px-4 py-3 outline-none focus:ring-2 focus:ring-[#002E7D] text-sm" placeholder="Pregunta por manuales o por tus datos...">
                    <button id="b" class="bg-[#002E7D] text-white p-3 md:p-4 rounded-2xl shadow-lg active:scale-95 transition-transform"><svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M14 5l7 7-7 7M5 12h16" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg></button>
                </div>
            </div>
        </div>

        <script>
            let history = [];
            const log = document.getElementById('log');
            const q = document.getElementById('q');
            const b = document.getElementById('b');
            const typing = document.getElementById('typing');
            const badge = document.getElementById('source-badge');

            function addMsg(text, isUser, sources = [], type = "") {
                const wrapper = document.createElement('div');
                wrapper.className = isUser ? "flex flex-row-reverse gap-3" : "flex gap-3";
                
                let sHtml = "";
                if (!isUser && sources.length > 0) {
                    sHtml = `
                        <div class="mt-3 pt-2 border-t border-gray-50 flex justify-end">
                            <div class="relative has-tooltip">
                                <div class="bg-blue-50 text-blue-600 rounded-full w-5 h-5 flex items-center justify-center text-[10px] cursor-help font-bold border border-blue-100">i</div>
                                <div class="tooltip bg-slate-800 text-white p-3 rounded-xl shadow-2xl z-50 text-[10px]">
                                    <p class="font-bold border-b border-white/10 mb-2 pb-1 text-blue-300">ORIGEN: ${type}</p>
                                    ${sources.map(s => `<div class="flex justify-between mb-1"><span>${s.name}</span><span class="text-blue-300 ml-2">${s.score}%</span></div>`).join('')}
                                </div>
                            </div>
                        </div>
                    `;
                }

                wrapper.innerHTML = `
                    <div class="w-8 h-8 rounded-full flex items-center justify-center font-bold text-[10px] shrink-0 border ${isUser?'bg-[#002E7D] text-white':'bg-white text-[#002E7D]'}">${isUser?'U':'AI'}</div>
                    <div class="${isUser?'bg-[#002E7D] text-white rounded-tr-none':'bg-white border text-gray-700 rounded-tl-none'} p-4 rounded-2xl shadow-sm text-sm leading-relaxed max-w-[85%]">
                        ${text.replace(/\\n/g, '<br>')}
                        ${sHtml}
                    </div>
                `;
                log.appendChild(wrapper);
                log.scrollTop = log.scrollHeight;
                history.push({role: isUser ? "user" : "assistant", content: text});
            }

            async function ask() {
                const val = q.value.trim(); if(!val) return;
                addMsg(val, true);
                q.value = '';
                typing.classList.remove('hidden');
                badge.classList.add('hidden');
                log.scrollTop = log.scrollHeight;

                try {
                    const res = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: val, history: history.slice(-6)})
                    });
                    const d = await res.json();
                    typing.classList.add('hidden');
                    
                    const sourceLabel = d.salesforce ? "SALESFORCE" : "RAG MANUALES";
                    badge.innerText = sourceLabel;
                    badge.classList.remove('hidden');

                    addMsg(d.answer, false, d.sources, sourceLabel);
                } catch(e) {
                    typing.classList.add('hidden');
                    addMsg("Error de conexión.", false);
                }
            }

            b.onclick = ask;
            q.onkeypress = (e) => { if(e.key === 'Enter') ask(); };
            window.onload = () => setTimeout(() => addMsg("¡Hola! Soy tu asistente híbrido. Puedes preguntarme por manuales o por tus datos personales de la cuenta.", false), 200);
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
