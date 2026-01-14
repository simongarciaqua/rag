from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
import re
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
        
        search_query = req.message
        if req.history:
            history_summary = "\n".join([f"{m.role}: {m.content}" for m in req.history[-3:]])
            rewrite_prompt = f"Basado en este historial:\n{history_summary}\n\nReescribe la pregunta '{req.message}' para que sea una búsqueda independiente en manuales. Di SOLO la búsqueda."
            search_query = model.generate_content(rewrite_prompt).text.strip()

        embed = genai.embed_content(model='models/text-embedding-004', content=search_query, task_type="retrieval_query")
        results = index.query(vector=embed['embedding'], top_k=5, include_metadata=True, namespace=pc_namespace)
        
        context_parts = []
        sources = []
        seen_files = set()
        for match in results.matches:
            if match.score > 0.4:
                text = match.metadata.get('text', '')
                f_name = match.metadata.get('file_name', 'Archivo')
                if text:
                    context_parts.append(text)
                    if f_name not in seen_files:
                        sources.append({"name": f_name, "score": round(match.score * 100, 1)})
                        seen_files.add(f_name)
        
        context = "\n\n".join(context_parts)
        gemini_history = []
        for m in req.history:
            gemini_history.append({"role": "user" if m.role == "user" else "model", "parts": [m.content]})
        
        chat_session = model.start_chat(history=gemini_history)
        system_instr = f"Eres un asistente de Aquaservice. Contexto manuales:\n{context}\n\nResponde a la pregunta del usuario. Si no está en el contexto, usa tu conocimiento general pero avisa."
        response = chat_session.send_message(f"{system_instr}\n\nPregunta: {req.message}")
        
        return {
            "answer": response.text if response.text else "No pude generar una respuesta.",
            "rag": len(context_parts) > 0,
            "sources": sources
        }
        
    except Exception as e:
        return {"answer": f"❌ Error: {str(e)}", "rag": False, "sources": []}

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
        <title>Aquaservice AI v3.5 Sticky</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            :root { --app-height: 100%; }
            body { 
                font-family: 'Outfit', sans-serif; 
                -webkit-tap-highlight-color: transparent;
                overscroll-behavior: none;
                height: 100vh;
                height: 100dvh;
            }
            .dot { animation: pulse 1.4s infinite; }
            @keyframes pulse { 0%, 100% { opacity: .2; } 50% { opacity: 1; } }
            
            .custom-scroll { -webkit-overflow-scrolling: touch; }
            
            /* Asegurar que el input no se cubra en iOS */
            .safe-bottom { padding-bottom: env(safe-area-inset-bottom); }
        </style>
    </head>
    <body class="bg-gray-100 flex flex-col items-center justify-center">
        
        <!-- App Container -->
        <div class="bg-white w-full md:max-w-2xl flex flex-col h-full md:h-[90vh] md:rounded-3xl shadow-2xl overflow-hidden relative">
            
            <!-- Header -->
            <div class="bg-[#002E7D] p-4 md:p-6 text-white flex justify-between items-center shrink-0 z-20">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 bg-white/10 rounded-xl flex items-center justify-center text-xl">💧</div>
                    <div>
                        <h1 class="text-base md:text-xl font-bold leading-tight">Aquaservice AI</h1>
                        <p class="text-blue-200 text-[10px] uppercase font-semibold">v3.5 Fixed-Bottom</p>
                    </div>
                </div>
            </div>

            <!-- Chat Space -->
            <div id="log" class="flex-1 overflow-y-auto p-4 md:p-6 space-y-4 bg-gray-50/30 custom-scroll">
                <!-- Messages -->
            </div>

            <!-- Bottom Area (Typing + Input) -->
            <div class="bg-white border-t border-gray-100 z-10 safe-bottom">
                
                <!-- Typing Ind -->
                <div id="typing" class="hidden px-4 py-2 flex gap-2 items-center text-[#002E7D] text-[10px] font-medium">
                    <span class="bg-blue-50 px-3 py-1 rounded-full border border-blue-100">
                        Escribiendo<span class="dot">.</span><span class="dot" style="animation-delay: 0.2s">.</span><span class="dot" style="animation-delay: 0.4s">.</span>
                    </span>
                </div>

                <!-- Input Group -->
                <div class="p-4 md:p-6 flex gap-2 md:gap-3 items-center">
                    <input id="q" type="text" enterkeyhint="send" autocomplete="off"
                        class="flex-1 bg-gray-50 border border-gray-200 rounded-2xl px-4 py-3 outline-none focus:ring-2 focus:ring-[#002E7D] text-sm transition-all" 
                        placeholder="Escribe tu mensaje...">
                    
                    <button id="b" class="bg-[#002E7D] text-white p-3 md:p-4 rounded-2xl shadow-lg active:scale-95 transition-transform shrink-0">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path d="M14 5l7 7-7 7M5 12h16" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </button>
                </div>
            </div>
        </div>

        <script>
            let history = [];
            const log = document.getElementById('log');
            const q = document.getElementById('q');
            const b = document.getElementById('b');
            const typing = document.getElementById('typing');

            function addMsg(text, isUser, sources = []) {
                const wrapper = document.createElement('div');
                wrapper.className = isUser ? "flex flex-row-reverse gap-3" : "flex gap-3";
                
                let sList = sources.length ? `
                    <div class="mt-2 text-[10px] text-blue-500 font-bold border-t border-blue-50 pt-2 flex flex-wrap gap-1">
                        ${sources.map(s => `<span class="bg-blue-50 px-2 py-0.5 rounded">📄 ${s.name}</span>`).join('')}
                    </div>` : "";

                wrapper.innerHTML = `
                    <div class="w-8 h-8 rounded-full flex items-center justify-center font-bold text-[10px] shrink-0 border ${isUser?'bg-[#002E7D] text-white border-transparent':'bg-white text-[#002E7D] border-gray-100'}">
                        ${isUser?'U':'AI'}
                    </div>
                    <div class="${isUser?'bg-[#002E7D] text-white rounded-tr-none shadow-blue-900/10':'bg-white border border-gray-100 text-gray-700 rounded-tl-none'} p-4 rounded-2xl shadow-sm text-sm leading-relaxed max-w-[85%]">
                        ${text.replace(/\\n/g, '<br>')}
                        ${sList}
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
                log.scrollTop = log.scrollHeight;

                try {
                    const res = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: val, history: history.slice(-6)})
                    });
                    const data = await res.json();
                    typing.classList.add('hidden');
                    addMsg(data.answer, false, data.sources);
                } catch(e) {
                    typing.classList.add('hidden');
                    addMsg("Error de conexión.", false);
                }
            }

            b.onclick = ask;
            q.onkeypress = (e) => { if(e.key === 'Enter') ask(); };

            window.onload = () => {
                setTimeout(() => addMsg("¡Hola! He ajustado la interfaz para que la barra de escritura esté siempre visible. ¿En qué puedo ayudarte?", false), 300);
            };
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
