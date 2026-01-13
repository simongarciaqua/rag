from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
import re

load_dotenv()

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

def clean_key(val):
    if not val: return ""
    # Eliminamos comillas accidentales y espacios
    return re.sub(r'[\s\n\r\t]', '', val).strip("'\" ")

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        # Cargamos variables dentro de la función para mayor robustez
        pc_key = clean_key(os.getenv('PINECONE_API_KEY'))
        index_name = clean_key(os.getenv('PINECONE_INDEX_NAME'))
        google_key = clean_key(os.getenv('GOOGLE_API_KEY'))

        if not pc_key or not index_name or not google_key:
            return {"answer": "❌ Error: Faltan variables de entorno en Vercel."}

        # Inicialización perezosa
        pc = Pinecone(api_key=pc_key)
        index = pc.Index(index_name)
        
        genai.configure(api_key=google_key)
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        
        # Generar embedding
        embed = genai.embed_content(
            model='models/text-embedding-004',
            content=req.message,
            task_type="retrieval_query"
        )
        
        # Query Pinecone
        results = index.query(vector=embed['embedding'], top_k=3, include_metadata=True)
        context = "\n\n".join([m.metadata.get('text', '') for m in results.matches])
        
        # Generar respuesta
        prompt = f"Contesta usando este contexto:\n\n{context}\n\nPregunta: {req.message}"
        response = model.generate_content(prompt)
        
        return {"answer": response.text}
        
    except Exception as e:
        msg = str(e)
        if "401" in msg or "Unauthorized" in msg:
            k = clean_key(os.getenv('PINECONE_API_KEY'))
            masked = f"{k[:10]}...{k[-5:]} (Len: {len(k)})"
            return {"answer": f"❌ ERROR 401: Pinecone rechaza la clave.\nClave en Vercel: {masked}\nCompara esto con tu local."}
        return {"answer": f"❌ Error: {msg}"}

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head><title>Aquaservice AI v2.6</title><script src="https://cdn.tailwindcss.com"></script></head>
    <body class="bg-slate-100 flex items-center justify-center h-screen">
        <div class="bg-white p-10 rounded-3xl shadow-2xl w-full max-w-lg">
            <h1 class="text-3xl font-bold text-blue-900 mb-6">Aquaservice AI v2.6</h1>
            <div id="log" class="h-64 overflow-y-auto mb-4 border-b p-2 text-sm space-y-2"></div>
            <div class="flex gap-2">
                <input id="q" class="flex-1 border p-3 rounded-full" placeholder="Escribe aquí...">
                <button id="b" class="bg-blue-900 text-white px-6 py-2 rounded-full">Enviar</button>
            </div>
        </div>
        <script>
            const log = document.getElementById('log');
            const q = document.getElementById('q');
            const b = document.getElementById('b');
            b.onclick = async () => {
                const val = q.value; if(!val) return;
                log.innerHTML += `<div><b>Tú:</b> ${val}</div>`;
                q.value = '';
                const r = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: val})
                });
                const d = await r.json();
                log.innerHTML += `<div class="bg-gray-50 p-2 rounded"><b>AI:</b> ${d.answer}</div>`;
                log.scrollTop = log.scrollHeight;
            };
        </script>
    </body>
    </html>
    """
