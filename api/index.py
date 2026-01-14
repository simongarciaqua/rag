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
        
        # 1. GENERAR QUERY DE BÚSQUEDA (CON MEMORIA)
        search_query = req.message
        if req.history:
            history_summary = "\n".join([f"{m.role}: {m.content}" for m in req.history[-3:]])
            rewrite_prompt = f"Basado en este historial:\n{history_summary}\n\nReescribe la pregunta '{req.message}' para que sea una búsqueda independiente en manuales. Di SOLO la búsqueda."
            search_query = model.generate_content(rewrite_prompt).text.strip()

        # 2. EMBEDDING Y BÚSQUEDA
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
        
        # 3. RESPUESTA FINAL
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
    <html>
    <head>
        <title>Aquaservice AI v3.3</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Outfit', sans-serif; }
            .dot { animation: pulse 1.4s infinite; }
            @keyframes pulse { 0%, 100% { opacity: .2; } 50% { opacity: 1; } }
        </style>
    </head>
    <body class="bg-slate-100 flex items-center justify-center min-h-screen p-4">
        <div class="bg-white rounded-3xl shadow-2xl w-full max-w-2xl flex flex-col h-[85vh] overflow-hidden">
            <div class="bg-[#002E7D] p-6 text-white flex justify-between items-center shrink-0">
                <div class="flex items-center gap-3">
                    <span class="text-2xl">💧</span>
                    <div>
                        <h1 class="text-xl font-bold">Aquaservice AI</h1>
                        <p class="text-blue-200 text-[10px] uppercase">Memoria v3.3</p>
                    </div>
                </div>
            </div>
            <div id="log" class="flex-1 overflow-y-auto p-6 space-y-6 bg-gray-50/50"></div>
            <div id="typing" class="hidden px-6 py-2 flex gap-2 items-center text-blue-500 text-xs">
                <span>Aquaservice está pensando</span>
                <span class="dot">.</span><span class="dot" style="animation-delay: 0.2s">.</span><span class="dot" style="animation-delay: 0.4s">.</span>
            </div>
            <div class="p-6 bg-white border-t flex gap-3 items-center shrink-0">
                <input id="q" class="flex-1 bg-gray-50 border border-gray-200 rounded-2xl px-5 py-4 outline-none focus:ring-2 focus:ring-blue-800" placeholder="Escribe tu duda...">
                <button id="b" class="bg-[#002E7D] text-white p-4 rounded-2xl shadow-lg">Enviar</button>
            </div>
        </div>

        <script>
            let history = [];
            const log = document.getElementById('log');
            const q = document.getElementById('q');
            const b = document.getElementById('b');
            const typing = document.getElementById('typing');

            function addMsg(text, isUser, sources = []) {
                const div = document.createElement('div');
                div.className = isUser ? "flex flex-row-reverse gap-3" : "flex gap-3";
                
                let sHtml = sources.length ? `
                    <div class="mt-2 text-[10px] text-blue-400 font-bold border-t pt-1">
                        Fuentes: ${sources.map(s => s.name).join(', ')}
                    </div>` : "";

                div.innerHTML = `
                    <div class="w-8 h-8 rounded-full flex items-center justify-center font-bold text-[10px] ${isUser?'bg-blue-900 text-white':'bg-blue-100 text-blue-800'}">${isUser?'U':'AI'}</div>
                    <div class="${isUser?'bg-blue-900 text-white':'bg-white border text-gray-700'} p-4 rounded-2xl shadow-sm text-sm max-w-[80%]">
                        ${text.replace(/\\n/g, '<br>')}
                        ${sHtml}
                    </div>
                `;
                log.appendChild(div);
                log.scrollTop = log.scrollHeight;
                history.push({role: isUser ? "user" : "assistant", content: text});
            }

            async function ask() {
                const val = q.value.trim(); if(!val) return;
                addMsg(val, true);
                q.value = '';
                typing.classList.remove('hidden');

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
                    addMsg("Error de conexión", false);
                }
            }

            b.onclick = ask;
            q.onkeypress = (e) => { if(e.key === 'Enter') ask(); };
            window.onload = () => addMsg("¡Hola! Ya estoy listo y con memoria activa. ¿En qué puedo ayudarte?", false);
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
