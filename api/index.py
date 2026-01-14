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
            return {"answer": "❌ Error: Faltan variables de entorno.", "rag": False, "sources": []}

        pc = Pinecone(api_key=pc_key)
        index = pc.Index(index_name)
        
        genai.configure(api_key=google_key)
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        
        # 1. Embedding
        embed = genai.embed_content(
            model='models/text-embedding-004',
            content=req.message,
            task_type="retrieval_query"
        )
        
        # 2. Pinecone Query
        results = index.query(
            vector=embed['embedding'], 
            top_k=5, 
            include_metadata=True,
            namespace=pc_namespace
        )
        
        context_parts = []
        seen_files = set()
        sources = []
        
        for match in results.matches:
            if match.score > 0.4:
                text = match.metadata.get('text', '')
                file_name = match.metadata.get('file_name', 'Documento')
                if text:
                    context_parts.append(text)
                    if file_name not in seen_files:
                        sources.append({"name": file_name, "score": round(match.score * 100, 1)})
                        seen_files.add(file_name)
        
        context = "\n\n".join(context_parts)
        rag_active = len(context_parts) > 0
        
        # 3. Prompt
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
            prompt = f"Contesta amablemente que no has encontrado información específica sobre '{req.message}' en los manuales de Aquaservice.\n\nPREGUNTA: {req.message}"

        response = model.generate_content(prompt)
        return {"answer": response.text, "rag": rag_active, "sources": sources}
        
    except Exception as e:
        return {"answer": f"❌ Error: {str(e)}", "rag": False, "sources": []}

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head>
        <title>Aquaservice AI v3.1</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Outfit', sans-serif; }
            .typing-dot {
                animation: typing 1.4s infinite ease-in-out both;
            }
            .typing-dot:nth-child(1) { animation-delay: -0.32s; }
            .typing-dot:nth-child(2) { animation-delay: -0.16s; }
            @keyframes typing {
                0%, 80%, 100% { transform: scale(0); }
                40% { transform: scale(1); }
            }
            .tooltip { visibility: hidden; opacity: 0; transition: opacity 0.3s; }
            .has-tooltip:hover .tooltip { visibility: visible; opacity: 1; }
        </style>
    </head>
    <body class="bg-slate-100 flex items-center justify-center min-h-screen p-4">
        <div class="bg-white rounded-3xl shadow-2xl w-full max-w-2xl flex flex-col h-[85vh] overflow-hidden">
            <!-- Header -->
            <div class="bg-[#002E7D] p-6 text-white flex justify-between items-center shrink-0">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                    </div>
                    <div>
                        <h1 class="text-xl font-bold">Aquaservice AI</h1>
                        <p class="text-blue-200 text-[10px] tracking-wider uppercase">Base de Conocimiento Activa</p>
                    </div>
                </div>
                <div id="status" class="text-[10px] bg-white/10 px-3 py-1 rounded-full border border-white/20 transition-all">
                    En línea
                </div>
            </div>
            
            <!-- Chat -->
            <div id="log" class="flex-1 overflow-y-auto p-6 space-y-6 bg-gray-50/50">
                <div class="flex gap-3">
                    <div class="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-800 font-bold text-[10px] shrink-0 shadow-sm border border-blue-200">AI</div>
                    <div class="bg-white p-4 rounded-2xl rounded-tl-none border border-gray-100 shadow-sm text-gray-700 text-sm max-w-[85%]">
                        ¡Hola! Soy tu asistente de Aquaservice. ¿Qué te gustaría consultar hoy?
                    </div>
                </div>
            </div>

            <!-- Input -->
            <div class="p-6 bg-white border-t border-gray-100 flex gap-3 items-center shrink-0">
                <div class="flex-1 relative">
                    <input id="q" class="w-full bg-gray-50 border border-gray-200 rounded-2xl px-5 py-4 outline-none focus:ring-2 focus:ring-blue-800 focus:bg-white transition-all text-sm pr-12" placeholder="Describe tu consulta...">
                </div>
                <button id="b" class="bg-[#002E7D] text-white p-4 rounded-2xl hover:scale-105 active:scale-95 transition-all shadow-lg hover:shadow-blue-900/20">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7-7 7M5 12h16"/></svg>
                </button>
            </div>
        </div>

        <script>
            const log = document.getElementById('log');
            const q = document.getElementById('q');
            const b = document.getElementById('b');
            const status = document.getElementById('status');

            function addMessage(text, isUser, sources = []) {
                const bubble = document.createElement('div');
                bubble.className = isUser ? "flex flex-row-reverse gap-3 animate-fade-in" : "flex gap-3 animate-fade-in";
                
                let sourceHtml = "";
                if (!isUser && sources.length > 0) {
                    sourceHtml = `
                        <div class="relative inline-block ml-2 has-tooltip">
                            <svg class="w-4 h-4 text-blue-400 cursor-help" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                            <div class="tooltip absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 bg-slate-800 text-white text-[10px] p-3 rounded-xl shadow-2xl z-50">
                                <p class="font-bold border-b border-white/10 pb-1 mb-2 uppercase tracking-tighter text-blue-300">Fuentes consultadas:</p>
                                ${sources.map(s => `<div class="flex justify-between items-center gap-2 mb-1"><span>${s.name}</span><span class="text-blue-300">${s.score}%</span></div>`).join('')}
                                <div class="absolute top-full left-1/2 -translate-x-1/2 border-8 border-transparent border-t-slate-800"></div>
                            </div>
                        </div>
                    `;
                }

                bubble.innerHTML = isUser ? `
                    <div class="w-8 h-8 rounded-full bg-[#002E7D] flex items-center justify-center text-white font-bold text-[10px] shrink-0 shadow-sm">U</div>
                    <div class="bg-[#002E7D] text-white p-4 rounded-2xl rounded-tr-none shadow-md max-w-[85%] text-sm">
                        ${text}
                    </div>
                ` : `
                    <div class="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-800 font-bold text-[10px] shrink-0 border border-blue-200 shadow-sm">AI</div>
                    <div class="bg-white p-4 rounded-2xl rounded-tl-none border border-gray-100 shadow-sm text-gray-700 text-sm max-w-[85%] leading-relaxed relative">
                        ${text}
                        <div class="mt-2 pt-2 border-t border-gray-50 flex items-center justify-end">
                            ${sourceHtml}
                        </div>
                    </div>
                `;
                
                log.appendChild(bubble);
                log.scrollTop = log.scrollHeight;
            }

            function showTyping() {
                const typing = document.createElement('div');
                typing.id = "typing-indicator";
                typing.className = "flex gap-3";
                typing.innerHTML = `
                    <div class="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-800 font-bold text-[10px] shrink-0 border border-blue-200 shadow-sm">AI</div>
                    <div class="bg-white p-4 rounded-2xl rounded-tl-none border border-gray-100 shadow-sm flex gap-1 items-center">
                        <div class="typing-dot w-1.5 h-1.5 bg-blue-400 rounded-full"></div>
                        <div class="typing-dot w-1.5 h-1.5 bg-blue-400 rounded-full"></div>
                        <div class="typing-dot w-1.5 h-1.5 bg-blue-400 rounded-full"></div>
                    </div>
                `;
                log.appendChild(typing);
                log.scrollTop = log.scrollHeight;
            }

            async function ask() {
                const val = q.value.trim(); if(!val) return;
                
                addMessage(val, true);
                q.value = '';
                status.innerHTML = "Generando respuesta...";
                status.className = "text-[10px] bg-blue-100 text-blue-700 px-3 py-1 rounded-full border border-blue-200 transition-all";
                
                showTyping();

                try {
                    const r = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: val})
                    });
                    const d = await r.json();
                    
                    document.getElementById('typing-indicator').remove();
                    
                    if(d.rag) {
                        status.innerHTML = "✨ Contexto recuperado";
                        status.className = "text-[10px] bg-emerald-100 text-emerald-700 px-3 py-1 rounded-full border border-emerald-200 transition-all";
                    } else {
                        status.innerHTML = "⚠️ Sin contexto específico";
                        status.className = "text-[10px] bg-amber-100 text-amber-700 px-3 py-1 rounded-full border border-amber-200 transition-all";
                    }

                    addMessage(d.answer, false, d.sources);
                } catch(e) {
                    if(document.getElementById('typing-indicator')) document.getElementById('typing-indicator').remove();
                    status.innerHTML = "❌ Error en conexión";
                    status.className = "text-[10px] bg-red-100 text-red-700 px-3 py-1 rounded-full border border-red-200 transition-all";
                    addMessage("Lo siento, hubo un error al procesar tu consulta.", false);
                }
            }

            b.onclick = ask;
            q.onkeypress = (e) => { if(e.key === 'Enter') ask(); };
        </script>
    </body>
    </html>
    """
