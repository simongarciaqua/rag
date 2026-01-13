from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
import traceback
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.api_core.exceptions

load_dotenv()

# --- CONFIG ---
class Config:
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
    PINECONE_NAMESPACE = os.getenv('PINECONE_NAMESPACE', 'default')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    EMBEDDING_MODEL = 'models/text-embedding-004'

    @classmethod
    def validate(cls):
        missing = []
        # Get ALL keys to see what's really happening
        all_keys = sorted(os.environ.keys())
        
        if not cls.PINECONE_API_KEY: missing.append("PINECONE_API_KEY")
        if not cls.PINECONE_INDEX_NAME: missing.append("PINECONE_INDEX_NAME")
        if not cls.GOOGLE_API_KEY: missing.append("GOOGLE_API_KEY")
        
        if missing:
            raise ValueError(
                f"Faltan: {', '.join(missing)}. "
                f"Variables totales en Vercel ({len(all_keys)}): {', '.join(all_keys)}"
            )

# --- APP ---
app = FastAPI()

# Definimos globales para reutilizar conexiones
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

@retry(
    retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def chat_with_gemini(chat_session, prompt):
    return chat_session.send_message(prompt)

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        index, model = get_resources()
        
        # 1. Embed
        res = genai.embed_content(
            model=Config.EMBEDDING_MODEL,
            content=req.message,
            task_type="retrieval_query"
        )
        query_vector = res['embedding']

        # 2. Search
        results = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True,
            namespace=Config.PINECONE_NAMESPACE
        )

        # 3. Context
        context = "\n\n".join([f"Fuente: {m.metadata.get('text', '')}" for m in results.matches])
        sources = [{"file_name": m.metadata.get('file_name', 'Doc'), "score": round(m.score, 3)} for m in results.matches]

        # 4. Answer
        prompt = f"Contesta a esta pregunta usando este CONTEXTO:\n\n{context}\n\nPREGUNTA: {req.message}"
        chat_session = model.start_chat(history=[])
        response = chat_with_gemini(chat_session, prompt)

        return {"answer": response.text, "sources": sources}

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {"answer": f"❌ Error: {str(e)}", "sources": []}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Aquaservice AI</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>body { background: #f3f4f6; font-family: sans-serif; }</style>
</head>
<body class="p-8 flex justify-center">
    <div class="w-full max-w-lg bg-white rounded-xl shadow-lg p-6">
        <h1 class="text-xl font-bold mb-4 text-blue-900">Aquaservice AI Chat</h1>
        <div id="chat" class="h-64 overflow-y-auto border-b mb-4 p-2 text-sm space-y-2"></div>
        <div class="flex gap-2">
            <input id="input" class="flex-1 border p-2 rounded" placeholder="Pregunta algo...">
            <button id="send" class="bg-blue-900 text-white px-4 py-2 rounded">Enviar</button>
        </div>
    </div>
    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        const btn = document.getElementById('send');

        btn.onclick = async () => {
            const msg = input.value;
            if(!msg) return;
            chat.innerHTML += `<div><b>Tú:</b> ${msg}</div>`;
            input.value = '';
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: msg})
                });
                const data = await res.json();
                chat.innerHTML += `<div class="bg-gray-100 p-2 rounded"><b>AI:</b> ${data.answer}</div>`;
            } catch (e) {
                chat.innerHTML += `<div class="text-red-500">Error de conexión</div>`;
            }
            chat.scrollTop = chat.scrollHeight;
        };
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
