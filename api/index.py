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
    if '=' in val: val = val.split('=')[-1]
    return re.sub(r'[\s\n\r\t]', '', val).strip("'\" ")

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        pc_key = clean_key(os.getenv('PINECONE_API_KEY'))
        index_name = clean_key(os.getenv('PINECONE_INDEX_NAME'))
        pc_namespace = clean_key(os.getenv('PINECONE_NAMESPACE', 'default'))
        google_key = clean_key(os.getenv('GOOGLE_API_KEY'))

        if not pc_key or not index_name or not google_key:
            return {"answer": "❌ Error: Faltan variables de entorno.", "rag": False}

        # Conexión Pinecone
        pc = Pinecone(api_key=pc_key)
        index = pc.Index(index_name)
        
        # Conexión Gemini
        genai.configure(api_key=google_key)
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        
        # 1. Generar embedding de la búsqueda con task_type="retrieval_query"
        embed = genai.embed_content(
            model='models/text-embedding-004',
            content=req.message,
            task_type="retrieval_query"
        )
        
        # 2. Query Pinecone usando el NAMESPACE explícito
        results = index.query(
            vector=embed['embedding'], 
            top_k=5, 
            include_metadata=True,
            namespace=pc_namespace
        )
        
        # Extraer fragmentos de texto
        context_parts = []
        for match in results.matches:
            if match.score > 0.4: # Filtro de relevancia simple
                text = match.metadata.get('text', '')
                if text:
                    context_parts.append(text)
        
        context = "\n\n".join(context_parts)
        rag_active = len(context_parts) > 0
        
        # 3. Generar respuesta con prompt estricto
        if rag_active:
            prompt = f"""Eres un asistente experto de Aquaservice. 
Usa EXCLUSIVAMENTE la siguiente información del contexto para responder. 
Si la información no está en el contexto, di que no lo sabes basándote en la documentación actual.

CONTEXTO:
{context}

PREGUNTA:
{req.message}

RESPUESTA:"""
        else:
            prompt = f"Contesta amablemente que no has encontrado información específica sobre '{req.message}' en los manuales de Aquaservice, pero intenta ayudar con lo que sepas como IA general mencionando que es una respuesta general.\n\nPREGUNTA: {req.message}"

        response = model.generate_content(prompt)
        return {"answer": response.text, "rag": rag_active}
        
    except Exception as e:
        msg = str(e)
        if "401" in msg or "Unauthorized" in msg:
            k = clean_key(os.getenv('PINECONE_API_KEY'))
            masked = f"{k[:10]}...{k[-5:]} (Len: {len(k)})"
            return {"answer": f"❌ ERROR 401 de Pinecone. Revisa la clave en Vercel.\nClave detectada: {masked}", "rag": False}
        return {"answer": f"❌ Error: {msg}", "rag": False}

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head>
        <title>Aquaservice AI v3.0</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            .rag-on { color: #10b981; }
            .rag-off { color: #f59e0b; }
        </style>
    </head>
    <body class="bg-slate-100 flex items-center justify-center min-h-screen p-4">
        <div class="bg-white rounded-3xl shadow-2xl w-full max-w-2xl flex flex-col h-[85vh] overflow-hidden">
            <div class="bg-[#002E7D] p-6 text-white flex justify-between items-center">
                <div>
                    <h1 class="text-2xl font-bold">Aquaservice AI</h1>
                    <p class="text-blue-200 text-xs">Sistema RAG Inteligente v3.0</p>
                </div>
                <div id="status" class="text-[10px] bg-white/10 px-3 py-1 rounded-full border border-white/20">
                    Esperando consulta...
                </div>
            </div>
            
            <div id="log" class="flex-1 overflow-y-auto p-6 space-y-4 bg-gray-50">
                <div class="flex gap-3">
                    <div class="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-800 font-bold text-xs shadow-sm">AI</div>
                    <div class="bg-white p-4 rounded-2xl rounded-tl-none border shadow-sm text-gray-700 max-w-[80%]">
                        Hola, soy el asistente de Aquaservice. ¿En qué puedo ayudarte hoy con respecto a nuestra base de conocimientos?
                    </div>
                </div>
            </div>

            <div class="p-4 bg-white border-t flex gap-2 items-center">
                <input id="q" class="flex-1 bg-gray-50 border border-gray-200 rounded-full px-5 py-3 outline-none focus:ring-2 focus:ring-blue-800 transition-all text-sm" placeholder="Escribe aquí tu pregunta...">
                <button id="b" class="bg-[#002E7D] text-white p-3 rounded-full hover:scale-105 active:scale-95 transition-all shadow-lg">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7-7 7M5 12h16"/></svg>
                </button>
            </div>
        </div>

        <script>
            const log = document.getElementById('log');
            const q = document.getElementById('q');
            const b = document.getElementById('b');
            const status = document.getElementById('status');

            async function ask() {
                const val = q.value.trim(); if(!val) return;
                
                // Add user message
                log.innerHTML += `
                    <div class="flex flex-row-reverse gap-3">
                        <div class="w-8 h-8 rounded-full bg-[#002E7D] flex items-center justify-center text-white font-bold text-xs shadow-sm">TÚ</div>
                        <div class="bg-[#002E7D] text-white p-4 rounded-2xl rounded-tr-none shadow-md max-w-[80%] text-sm">
                            ${val}
                        </div>
                    </div>
                `;
                
                q.value = '';
                log.scrollTop = log.scrollHeight;
                status.innerHTML = "Consultando base de datos...";

                try {
                    const r = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: val})
                    });
                    const d = await r.json();
                    
                    // Update status
                    if(d.rag) {
                        status.innerHTML = "✨ Contexto recuperado";
                        status.className = "text-[10px] bg-emerald-100 text-emerald-700 px-3 py-1 rounded-full border border-emerald-200";
                    } else {
                        status.innerHTML = "⚠️ Sin contexto específico";
                        status.className = "text-[10px] bg-amber-100 text-amber-700 px-3 py-1 rounded-full border border-amber-200";
                    }

                    // Add AI answer
                    log.innerHTML += `
                        <div class="flex gap-3">
                            <div class="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-800 font-bold text-xs shadow-sm">AI</div>
                            <div class="bg-white p-4 rounded-2xl rounded-tl-none border shadow-sm text-gray-700 max-w-[80%] text-sm">
                                ${d.answer}
                            </div>
                        </div>
                    `;
                } catch(e) {
                    status.innerHTML = "❌ Error en conexión";
                    status.className = "text-[10px] bg-red-100 text-red-700 px-3 py-1 rounded-full border border-red-200";
                }
                
                log.scrollTop = log.scrollHeight;
            }

            b.onclick = ask;
            q.onkeypress = (e) => { if(e.key === 'Enter') ask(); };
        </script>
    </body>
    </html>
    """
