from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
import re
from typing import List, Dict

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
        
        # 1. GENERAR QUERY DE BÚSQUEDA (CON MEMORIA)
        # Si hay historial, pedimos a Gemini que resuma la intención de búsqueda
        search_query = req.message
        if req.history:
            history_text = "\\n".join([f"{m.role}: {m.content}" for m in req.history[-3:]])
            rewrite_prompt = f"Basado en este historial de chat:\\n{history_text}\\n\\nConvierte esta última pregunta del usuario en una frase de búsqueda independiente para una base de datos de manuales (RAG): '{req.message}'. Responde SOLO con la frase de búsqueda."
            search_query_res = model.generate_content(rewrite_prompt)
            search_query = search_query_res.text.strip()

        # 2. EMBEDDING Y BÚSQUEDA
        embed = genai.embed_content(model='models/text-embedding-004', content=search_query, task_type="retrieval_query")
        results = index.query(vector=embed['embedding'], top_k=5, include_metadata=True, namespace=pc_namespace)
        
        context_parts = []
        sources = []
        seen_files = set()
        for match in results.matches:
            if match.score > 0.4:
                text = match.metadata.get('text', '')
                f_name = match.metadata.get('file_name', 'Doc')
                if text:
                    context_parts.append(text)
                    if f_name not in seen_files:
                        sources.append({"name": f_name, "score": round(match.score * 100, 1)})
                        seen_files.add(f_name)
        
        context = "\\n\\n".join(context_parts)
        rag_active = len(context_parts) > 0
        
        # 3. RESPUESTA FINAL (CON HISTORIAL)
        gemini_history = []
        for m in req.history:
            gemini_history.append({"role": "user" if m.role == "user" else "model", "parts": [m.content]})
        
        chat_session = model.start_chat(history=gemini_history)
        
        system_instr = f"Eres un asistente de Aquaservice. Usa este CONTEXTO EXTRAÍDO DE LOS MANUALES para responder:\\n{context}\\n\\nSi el contexto no tiene la respuesta, usa tu conocimiento general pero indica que no está en los manuales oficiales."
        
        final_response = chat_session.send_message(f"{system_instr}\\n\\nUSUARIO: {req.message}")
        
        return {"answer": final_response.text, "rag": rag_active, "sources": sources}
        
    except Exception as e:
        return {"answer": f"❌ Error: {str(e)}", "rag": False, "sources": []}

@app.get("/", response_class=HTMLResponse)
async def home():
    return \"\"\"
    <html>
    <head>
        <title>Aquaservice AI v3.2 (Con Memoria)</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Outfit', sans-serif; }
            .typing-dot { animation: typing 1.4s infinite ease-in-out both; }
            .typing-dot:nth-child(1) { animation-delay: -0.32s; }
            .typing-dot:nth-child(2) { animation-delay: -0.16s; }
            @keyframes typing { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
            .tooltip { visibility: hidden; opacity: 0; transition: opacity 0.3s; }
            .has-tooltip:hover .tooltip { visibility: visible; opacity: 1; }
        </style>
    </head>
    <body class="bg-slate-100 flex items-center justify-center min-h-screen p-4">
        <div class="bg-white rounded-3xl shadow-2xl w-full max-w-2xl flex flex-col h-[85vh] overflow-hidden">
            <div class="bg-[#002E7D] p-6 text-white flex justify-between items-center shrink-0">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center">⚡</div>
                    <div>
                        <h1 class="text-xl font-bold">Aquaservice AI</h1>
                        <p class="text-blue-200 text-[10px] uppercase tracking-widest">Memoria Inteligente v3.2</p>
                    </div>
                </div>
                <div id="status" class="text-[10px] bg-white/10 px-3 py-1 rounded-full border border-white/20">En línea</div>
            </div>
            <div id="log" class="flex-1 overflow-y-auto p-6 space-y-6 bg-gray-50/50"></div>
            <div class="p-6 bg-white border-t border-gray-100 flex gap-3 items-center">
                <input id="q" class="flex-1 bg-gray-50 border border-gray-200 rounded-2xl px-5 py-4 outline-none focus:ring-2 focus:ring-blue-800 text-sm" placeholder="Escribe tu mensaje...">
                <button id="b" class="bg-[#002E7D] text-white p-4 rounded-2xl shadow-lg hover:scale-105 transition-all">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M14 5l7 7-7 7M5 12h16" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
                </button>
            </div>
        </div>
        <script>
            let chatHistory = [];
            const log = document.getElementById('log');
            const q = document.getElementById('q');
            const b = document.getElementById('b');
            const status = document.getElementById('status');

            function addMessage(text, isUser, sources = []) {
                const bubble = document.createElement('div');
                bubble.className = isUser ? "flex flex-row-reverse gap-3" : "flex gap-3";
                
                let sourceHtml = "";
                if (!isUser && sources.length > 0) {
                    sourceHtml = `
                        <div class="relative inline-block ml-2 has-tooltip">
                            <span class="text-blue-400 cursor-help text-xs">ⓘ</span>
                            <div class="tooltip absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 bg-slate-800 text-white text-[10px] p-3 rounded-xl shadow-2xl z-50">
                                <p class="font-bold border-b border-white/10 mb-1">FUENTES:</p>
                                ${sources.map(s => `<div class="flex justify-between"><span>${s.name}</span><span>${s.score}%</span></div>`).join('')}
                            </div>
                        </div>
                    `;
                }

                bubble.innerHTML = isUser ? `
                    <div class="w-8 h-8 rounded-full bg-[#002E7D] flex items-center justify-center text-white font-bold text-[10px]">U</div>
                    <div class="bg-[#002E7D] text-white p-4 rounded-2xl rounded-tr-none shadow-md max-w-[85%] text-sm">${text}</div>
                ` : `
                    <div class="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-800 font-bold text-[10px] border border-blue-200">AI</div>
                    <div class="bg-white p-4 rounded-2xl rounded-tl-none border border-gray-100 shadow-sm text-gray-700 text-sm max-w-[85%]">
                        ${text.replace(/\\n/g, '<br>')}
                        <div class="mt-2 text-right">${sourceHtml}</div>
                    </div>
                `;
                log.appendChild(bubble);
                log.scrollTop = log.scrollHeight;
                if(text) chatHistory.push({role: isUser ? "user" : "assistant", content: text});
            }

            async function ask() {
                const val = q.value.trim(); if(!val) return;
                addMessage(val, true);
                q.value = '';
                const typing = document.createElement('div');
                typing.id = "typing";
                typing.className = "flex gap-3";
                typing.innerHTML = '<div class="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-800 font-bold text-[10px]">AI</div><div class="bg-white p-4 rounded-2xl text-blue-400 flex gap-1"><div class="typing-dot w-1.5 h-1.5 bg-current rounded-full"></div><div class="typing-dot w-1.5 h-1.5 bg-current rounded-full"></div><div class="typing-dot w-1.5 h-1.5 bg-current rounded-full"></div></div>';
                log.appendChild(typing);
                log.scrollTop = log.scrollHeight;

                const r = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: val, history: chatHistory.slice(-6)})
                });
                const d = await r.json();
                document.getElementById('typing').remove();
                addMessage(d.answer, false, d.sources);
            }

            b.onclick = ask;
            q.onkeypress = (e) => { if(e.key === 'Enter') ask(); };
            window.onload = () => addMessage("¡Hola! Ya tengo memoria. Háblame de algo y luego hazme preguntas de seguimiento.", false);
        </script>
    </body>
    </html>
    \"\"\"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
