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
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', '').strip()
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', '').strip()
    PINECONE_NAMESPACE = os.getenv('PINECONE_NAMESPACE', 'default').strip()
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '').strip()
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
        # Detectamos errores comunes para dar mensajes más amigables
        error_msg = str(e)
        if "Unauthorized" in error_msg or "API Key" in error_msg:
            friendly_err = "❌ Error de Autorización: Tu clave de Pinecone es inválida. Revísala en Vercel."
        elif "ResourceExhausted" in error_msg:
            friendly_err = "⚠️ Límite excedido: El modelo de Google Gemini está saturado. Reintentando..."
        else:
            friendly_err = f"❌ Error: {error_msg}"
            
        return {"answer": friendly_err, "sources": []}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aquaservice RAG</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Outfit', sans-serif; background-color: #f3f4f6; }
        .bg-aquaservice { background-color: #002E7D; }
        .typing-dot { animation: typing 1.4s infinite ease-in-out both; }
        @keyframes typing { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
    </style>
</head>
<body class="h-screen flex flex-col items-center justify-center p-4">

    <div class="w-full max-w-2xl bg-white rounded-3xl shadow-2xl overflow-hidden flex flex-col h-[85vh]">
        <!-- Header -->
        <div class="bg-aquaservice p-6 flex items-center justify-between">
            <div class="flex items-center gap-3 text-white">
                <div class="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/></svg>
                </div>
                <div>
                    <h1 class="font-semibold text-xl">Aquaservice AI</h1>
                    <p class="text-blue-200 text-xs">Asistente Virtual Inteligente</p>
                </div>
            </div>
        </div>

        <!-- Chat Area -->
        <div id="chat" class="flex-1 p-6 overflow-y-auto space-y-4 bg-gray-50">
            <div class="flex gap-3">
                <div class="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 font-bold text-xs">AI</div>
                <div class="bg-white p-3 rounded-2xl rounded-tl-none shadow-sm text-sm text-gray-700 border border-gray-100">
                    ¡Hola! Soy el asistente de Aquaservice. ¿Qué información necesitas?
                </div>
            </div>
        </div>

        <!-- Input -->
        <div class="p-4 bg-white border-t border-gray-100">
            <div class="flex gap-2">
                <input type="text" id="input" 
                    class="flex-1 px-5 py-3 bg-gray-50 border border-gray-200 rounded-full text-sm outline-none focus:ring-2 focus:ring-blue-500/20"
                    placeholder="Escribe tu pregunta...">
                <button id="send" class="p-3 bg-aquaservice text-white rounded-full hover:bg-blue-800 transition-colors shadow-lg">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7-7 7M5 12h16"/></svg>
                </button>
            </div>
        </div>
    </div>

    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        const btn = document.getElementById('send');

        function appendMsg(who, text, isAi = false) {
            const div = document.createElement('div');
            div.className = `flex gap-3 ${who === 'Tú' ? 'flex-row-reverse' : ''}`;
            div.innerHTML = `
                <div class="w-8 h-8 rounded-full flex items-center justify-center font-bold text-xs ${isAi ? 'bg-blue-100 text-blue-600' : 'bg-gray-200 text-gray-600'}">${who === 'AI' ? 'AI' : 'U'}</div>
                <div class="${isAi ? 'bg-white text-gray-700 border border-gray-100' : 'bg-blue-900 text-white'} p-3 rounded-2xl ${isAi ? 'rounded-tl-none' : 'rounded-tr-none'} shadow-sm text-sm max-w-[80%]">
                    ${text}
                </div>
            `;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        async function talk() {
            const msg = input.value.trim();
            if(!msg) return;
            appendMsg('Tú', msg);
            input.value = '';

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: msg})
                });
                const data = await res.json();
                appendMsg('AI', data.answer, true);
            } catch (e) {
                appendMsg('AI', '❌ Error de conexión al servidor.', true);
            }
        }

        btn.onclick = talk;
        input.onkeypress = (e) => { if(e.key === 'Enter') talk(); };
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
