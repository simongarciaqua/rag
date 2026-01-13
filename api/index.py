from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
import traceback
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.api_core.exceptions

load_dotenv()

# --- CONFIG ---
class Config:
    @staticmethod
    def _clean(val, name=""):
        if not val: 
            print(f"DEBUG: Variable {name} esta VACIA")
            return ""
        clean_val = re.sub(r'[^\x21-\x7E]', '', val).strip("'\" ")
        # LOG DE SEGURIDAD PARA VER QUÉ LLEGA A VERCEL
        print(f"DEBUG: Variable {name} cargada: {clean_val[:4]}...{clean_val[-4:]} (Longitud: {len(clean_val)})")
        return clean_val

    PINECONE_API_KEY = _clean(os.getenv('PINECONE_API_KEY'), "PINECONE_API_KEY")
    PINECONE_INDEX_NAME = _clean(os.getenv('PINECONE_INDEX_NAME'), "PINECONE_INDEX_NAME")
    PINECONE_NAMESPACE = _clean(os.getenv('PINECONE_NAMESPACE', 'default'), "PINECONE_NAMESPACE")
    GOOGLE_API_KEY = _clean(os.getenv('GOOGLE_API_KEY'), "GOOGLE_API_KEY")
    EMBEDDING_MODEL = 'models/text-embedding-004'

    @classmethod
    def validate(cls):
        missing = []
        if not cls.PINECONE_API_KEY: missing.append("PINECONE_API_KEY")
        if not cls.PINECONE_INDEX_NAME: missing.append("PINECONE_INDEX_NAME")
        if not cls.GOOGLE_API_KEY: missing.append("GOOGLE_API_KEY")
        if missing:
            raise ValueError(f"Faltan variables: {', '.join(missing)}")

# --- APP ---
app = FastAPI()

_pc_index = None
_chat_model = None

def get_resources():
    global _pc_index, _chat_model
    if _pc_index is None:
        Config.validate()
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
        sources = [{"file_name": m.metadata.get('file_name', 'Doc'), "score": round(m.score, 3)} for m in results.matches]
        prompt = f"Contesta a esta pregunta usando este CONTEXTO:\n\n{context}\n\nPREGUNTA: {req.message}"
        chat_session = model.start_chat(history=[])
        response = chat_with_gemini(chat_session, prompt)
        return {"answer": response.text, "sources": sources}
    except Exception as e:
        print(f"ERROR: {str(e)}")
        error_msg = str(e)
        if "Unauthorized" in error_msg or "API Key" in error_msg:
            pk = Config.PINECONE_API_KEY
            friendly_err = f"❌ Error de Autorización (Pinecone). La clave que ve Vercel es: {pk[:4]}...{pk[-4:]} (Longitud: {len(pk)}). Revísala."
        else:
            friendly_err = f"❌ Error: {error_msg}"
        return {"answer": friendly_err, "sources": []}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return """
<!DOCTYPE html>
<html>
<head><title>Aquaservice AI</title><script src="https://cdn.tailwindcss.com"></script></head>
<body class="p-8 flex justify-center bg-gray-100">
    <div class="w-full max-w-lg bg-white rounded-xl shadow-lg p-6">
        <h1 class="text-xl font-bold mb-4 text-blue-900 text-center">Aquaservice AI Chat</h1>
        <div id="chat" class="h-80 overflow-y-auto mb-4 p-4 text-sm space-y-3 bg-gray-50 rounded-lg border"></div>
        <div class="flex gap-2">
            <input id="input" class="flex-1 border p-3 rounded-full outline-none focus:ring-2 focus:ring-blue-500" placeholder="Pregunta algo...">
            <button id="send" class="bg-blue-900 text-white px-6 py-2 rounded-full hover:bg-blue-800 transition-colors">Enviar</button>
        </div>
    </div>
    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        const btn = document.getElementById('send');
        async function talk() {
            const msg = input.value; if(!msg) return;
            chat.innerHTML += `<div class="text-right"><span class="bg-blue-900 text-white p-2 rounded-lg inline-block">${msg}</span></div>`;
            input.value = '';
            try {
                const res = await fetch('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({message: msg})});
                const data = await res.json();
                chat.innerHTML += `<div class="text-left"><span class="bg-white border p-2 rounded-lg inline-block text-gray-800">${data.answer}</span></div>`;
            } catch (e) { chat.innerHTML += `<div class="text-red-500 text-center">Error de conexión</div>`; }
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
