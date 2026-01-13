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
        prompt = f"Contesta usando este CONTEXTO:\n\n{context}\n\nPREGUNTA: {req.message}"
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
<head><title>Chat Aquaservice</title><script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-blue-50 h-screen flex items-center justify-center p-4">
    <div class="w-full max-w-xl bg-white rounded-2xl shadow-xl flex flex-col h-[80vh]">
        <div class="bg-blue-900 p-4 text-white font-bold rounded-t-2xl">Aquaservice AI</div>
        <div id="chat" class="flex-1 overflow-y-auto p-4 space-y-4 text-sm"></div>
        <div class="p-4 border-t flex gap-2">
            <input id="input" class="flex-1 border rounded-full px-4 outline-none focus:ring-2 focus:ring-blue-900" placeholder="Escribe tu duda...">
            <button id="send" class="bg-blue-900 text-white p-2 rounded-full px-6">Enviar</button>
        </div>
    </div>
    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        const btn = document.getElementById('send');
        async function talk() {
            const m = input.value; if(!m) return;
            chat.innerHTML += `<div class="text-right"><span class="bg-blue-900 text-white p-2 rounded-lg inline-block">${m}</span></div>`;
            input.value = '';
            const res = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message:m})});
            const d = await res.json();
            chat.innerHTML += `<div class="text-left"><span class="bg-gray-100 p-2 rounded-lg inline-block text-gray-800">${d.answer}</span></div>`;
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
