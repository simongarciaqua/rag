from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.api_core.exceptions

load_dotenv()

class Config:
    @staticmethod
    def _clean(val):
        if not val: return ""
        # Limpieza profunda de caracteres invisibles y espacios
        return re.sub(r'[^\x21-\x7E]', '', val).strip("'\" ")

    PINECONE_API_KEY = _clean(os.getenv('PINECONE_API_KEY'))
    PINECONE_INDEX_NAME = _clean(os.getenv('PINECONE_INDEX_NAME'))
    PINECONE_NAMESPACE = _clean(os.getenv('PINECONE_NAMESPACE', 'default'))
    GOOGLE_API_KEY = _clean(os.getenv('GOOGLE_API_KEY'))
    EMBEDDING_MODEL = 'models/text-embedding-004'

app = FastAPI()

_pc_index = None
_chat_model = None

def get_resources():
    global _pc_index, _chat_model
    if _pc_index is None:
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        _pc_index = pc.Index(Config.PINECONE_INDEX_NAME)
    if _chat_model is None:
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        _chat_model = genai.GenerativeModel('models/gemini-2.0-flash')
    return _pc_index, _chat_model

@retry(retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted), stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def chat_with_gemini(chat_session, prompt):
    return chat_session.send_message(prompt)

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        index, model = get_resources()
        res = genai.embed_content(model=Config.EMBEDDING_MODEL, content=req.message, task_type="retrieval_query")
        query_vector = res['embedding']
        results = index.query(vector=query_vector, top_k=5, include_metadata=True, namespace=Config.PINECONE_NAMESPACE)
        context = "\n\n".join([f"Fuente: {m.metadata.get('text', '')}" for m in results.matches])
        prompt = f"Eres un asistente de Aquaservice. Contesta usando este CONTEXTO:\n\n{context}\n\nPREGUNTA: {req.message}"
        chat_session = model.start_chat(history=[])
        response = chat_with_gemini(chat_session, prompt)
        return {"answer": response.text}
    except Exception as e:
        return {"answer": f"❌ Error: {str(e)}"}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return """
<!DOCTYPE html>
<html>
<head><title>Aquaservice AI v2</title><script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-slate-100 h-screen flex items-center justify-center p-4">
    <div class="w-full max-w-xl bg-white rounded-3xl shadow-2xl flex flex-col h-[85vh] border border-gray-100">
        <div class="bg-[#002E7D] p-6 text-white rounded-t-3xl flex justify-between items-center">
            <div>
                <h1 class="font-bold text-xl">Aquaservice AI</h1>
                <p class="text-blue-200 text-xs">Versión Desplegada v2</p>
            </div>
            <div class="w-3 h-3 bg-green-400 rounded-full shadow-[0_0_10px_#4ade80]"></div>
        </div>
        <div id="chat" class="flex-1 overflow-y-auto p-6 space-y-4 text-sm bg-gray-50">
            <div class="flex gap-3">
                <div class="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-800 font-bold text-[10px]">AI</div>
                <div class="bg-white p-3 rounded-2xl rounded-tl-none border shadow-sm text-gray-700">¡Bienvenido! Ya estoy conectado correctamente. ¿Qué quieres saber?</div>
            </div>
        </div>
        <div class="p-4 bg-white border-t flex gap-2">
            <input id="input" class="flex-1 bg-gray-50 border rounded-full px-5 py-3 outline-none focus:ring-2 focus:ring-blue-800 transition-all" placeholder="Escribe tu mensaje...">
            <button id="send" class="bg-[#002E7D] text-white p-3 rounded-full hover:scale-105 active:scale-95 transition-all">
                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7-7 7M5 12h16"/></svg>
            </button>
        </div>
    </div>
    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        const btn = document.getElementById('send');
        async function talk() {
            const m = input.value.trim(); if(!m) return;
            chat.innerHTML += `<div class="flex flex-row-reverse gap-3"><div class="w-8 h-8 rounded-full bg-[#002E7D] flex items-center justify-center text-white font-bold text-[10px]">U</div><div class="bg-[#002E7D] text-white p-3 rounded-2xl rounded-tr-none shadow-md inline-block max-w-[80%]">${m}</div></div>`;
            input.value = ''; chat.scrollTop = chat.scrollHeight;
            const res = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message:m})});
            const d = await res.json();
            chat.innerHTML += `<div class="flex gap-3"><div class="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-800 font-bold text-[10px]">AI</div><div class="bg-white p-3 rounded-2xl rounded-tl-none border shadow-sm text-gray-700 inline-block max-w-[80%]">${d.answer}</div></div>`;
            chat.scrollTop = chat.scrollHeight;
        }
        btn.onclick = talk; input.onkeypress = (e) => { if(e.key === 'Enter') talk(); };
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
