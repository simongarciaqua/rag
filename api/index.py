from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
import re

load_dotenv()

def clean_env(val):
    if not val: return ""
    # Eliminamos CUALQUIER cosa que no sea un carácter de clave estándar
    return re.sub(r'[\s\n\r\t]', '', val).strip("'\" ")

class Config:
    PINECONE_API_KEY = clean_env(os.getenv('PINECONE_API_KEY'))
    PINECONE_INDEX_NAME = clean_env(os.getenv('PINECONE_INDEX_NAME'))
    GOOGLE_API_KEY = clean_env(os.getenv('GOOGLE_API_KEY'))

app = FastAPI()

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        # DIAGNÓSTICO EN VIVO
        pk = Config.PINECONE_API_KEY
        masked = f"{pk[:8]}...{pk[-4:]}" if len(pk) > 12 else "ERROR: Clave demasiado corta"
        
        print(f"DEBUG: Intentando conectar con clave {masked} (Longitud: {len(pk)})")
        
        pc = Pinecone(api_key=pk)
        index = pc.Index(Config.PINECONE_INDEX_NAME)
        
        # Prueba rápida: listar índices (aquí es donde suele dar el 401)
        idx_list = [i.name for i in pc.list_indexes()]
        
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        
        res = genai.embed_content(model='models/text-embedding-004', content=req.message, task_type="retrieval_query")
        results = index.query(vector=res['embedding'], top_k=3, include_metadata=True)
        
        context = "\n\n".join([m.metadata.get('text', '') for m in results.matches])
        response = model.generate_content(f"Contexto: {context}\n\nPregunta: {req.message}")
        
        return {"answer": response.text}
        
    except Exception as e:
        err_str = str(e).lower()
        pk = Config.PINECONE_API_KEY
        masked = f"{pk[:8]}...{pk[-4:]}" if len(pk) > 12 else "INVALIDA"
        
        if "unauthorized" in err_str or "401" in err_str:
            return {"answer": f"❌ PINE_401: Pinecone rechaza esta clave.\n• Tu clave en Vercel es: {masked}\n• Longitud real: {len(pk)} caracteres.\n• Compara esto con tu .env local paso a paso."}
        return {"answer": f"❌ Error: {str(e)}"}

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return """
<!DOCTYPE html>
<html>
<head><title>Aquaservice v2.4</title><script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-gray-100 h-screen flex items-center justify-center">
    <div class="bg-white p-8 rounded-3xl shadow-xl w-full max-w-lg border-t-8 border-blue-900">
        <h1 class="text-2xl font-bold text-blue-900 mb-2">Aquaservice AI</h1>
        <p class="text-xs text-gray-400 mb-6 font-mono">Build v2.4 (Security Clean)</p>
        <div id="chat" class="h-64 overflow-y-auto mb-4 space-y-2 text-sm"></div>
        <div class="flex gap-2">
            <input id="input" class="flex-1 border p-3 rounded-full outline-none" placeholder="Pregunta algo...">
            <button id="send" class="bg-blue-900 text-white px-6 py-2 rounded-full">Enviar</button>
        </div>
    </div>
    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        const btn = document.getElementById('send');
        btn.onclick = async () => {
            const m = input.value; if(!m) return;
            chat.innerHTML += `<div class="text-right text-blue-900"><b>Tú:</b> ${m}</div>`;
            input.value = '';
            const res = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message:m})});
            const d = await res.json();
            chat.innerHTML += `<div class="bg-gray-50 p-3 rounded-xl"><b>AI:</b> ${d.answer}</div>`;
            chat.scrollTop = chat.scrollHeight;
        };
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
