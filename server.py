from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
import traceback
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.api_core.exceptions

# Re-use our existing modules
from rag_sync.src.config import Config
from rag_sync.src.ingestion import Processor

load_dotenv()

app = FastAPI()

# --- SETUP RAG ---
Config.validate()
pc = Pinecone(api_key=Config.PINECONE_API_KEY)
index = pc.Index(Config.PINECONE_INDEX_NAME)
processor = Processor(Config.GOOGLE_API_KEY)
genai.configure(api_key=Config.GOOGLE_API_KEY)

# Using gemini-2.0-flash as confirmed available
chat_model = genai.GenerativeModel('models/gemini-2.0-flash') 

# Define retry logic specifically for Rate Limits (429)
# Wait exponentially between 4s and 60s, trying up to 5 times.
@retry(
    retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
def generate_content_with_retry(chat_session, prompt):
    print("  -> Calling Gemini API...")
    return chat_session.send_message(prompt)

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        print(f"Received query: {req.message}")

        # 1. Embed Query
        embeddings = processor.embed_batch([req.message])
        if not embeddings:
            print("ERROR: processor.embed_batch returned empty or None")
            raise HTTPException(status_code=500, detail="Failed to embed query")
        
        query_vector = embeddings[0]
        print(f"Embedding generated. Vector len: {len(query_vector)}")

        # 2. Search Pinecone
        print(f"Querying Pinecone Index: {Config.PINECONE_INDEX_NAME}, Namespace: {Config.PINECONE_NAMESPACE}")
        results = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True,
            namespace=Config.PINECONE_NAMESPACE
        )
        print(f"Pinecone results found: {len(results.matches)}")

        # 3. Build Context
        context_parts = []
        sources = []
        
        for match in results.matches:
            # Handle potential missing keys gracefully
            metadata = match.metadata or {}
            text = metadata.get('text', '')
            score = match.score
            file_name = metadata.get('file_name', 'Unknown')
            
            context_parts.append(f"Source ({file_name}): {text}")
            
            # Simple deduplication for sources info
            if not any(s['file_name'] == file_name for s in sources):
                sources.append({
                    "file_name": file_name,
                    "score": round(score, 3)
                })

        context_str = "\n\n".join(context_parts)
        print("Context built.")
        
        # 4. Generate Answer
        system_instruction = f"""You are a helpful assistant for Aquaservice.
        Use the following CONTEXT to answer the user's question.
        If the answer is not in the context, just say you don't know politely.
        
        CONTEXT:
        {context_str}
        """
        
        generated_text = ""
        try:
            print("Sending to Gemini Chat Model (with retry)...")
            chat_session = chat_model.start_chat(history=[])
            
            # Use the retry wrapper
            prompt = f"{system_instruction}\n\nUSER QUESTION: {req.message}"
            response = generate_content_with_retry(chat_session, prompt)
            generated_text = response.text
            print("Gemini response received.")
            
        except Exception as llm_error:
            print(f"LLM Generation failed: {llm_error}")
            # Fallback: Return the context directly if LLM fails
            generated_text = (
                "⚠️ **Aviso de Sistema**: No he podido contactar con el modelo de generación (posiblemente por límites de cuota de Google API).\n\n"
                "Sin embargo, **el sistema RAG funciona correctamente**. Aquí tienes la información relevante que he encontrado en tus documentos:\n\n"
                f"{context_str}\n\n"
                "*(Intenta de nuevo más tarde para ver la respuesta redactada)*"
            )

        return {
            "answer": generated_text,
            "sources": sources
        }

    except Exception as e:
        traceback.print_exc()
        print(f"Error (Global): {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- FRONTEND ---
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aquaservice RAG</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Outfit', sans-serif; background-color: #f3f4f6; }
        .chat-container { height: calc(100vh - 180px); }
        .aquaservice-blue { color: #002E7D; }
        .bg-aquaservice { background-color: #002E7D; }
        .message-bubble { max-width: 80%; }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
        
        /* Info Icon Tooltip */
        .info-icon:hover + .tooltip { display: block; }
        .tooltip { display: none; }
        
        /* Loading Dots */
        .typing-dot {
            animation: typing 1.4s infinite ease-in-out both;
        }
        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes typing {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body class="h-screen flex flex-col items-center justify-center p-4">

    <!-- Card -->
    <div class="w-full max-w-2xl bg-white rounded-3xl shadow-2xl overflow-hidden flex flex-col h-[85vh]">
        
        <!-- Header -->
        <div class="bg-aquaservice p-6 flex items-center justify-between">
            <div class="flex items-center gap-3">
                <div class="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center">
                    <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                </div>
                <div>
                    <h1 class="text-white font-semibold text-xl">Aquaservice AI</h1>
                    <p class="text-blue-200 text-xs">Conectado a Base de Conocimiento</p>
                </div>
            </div>
            <div class="w-3 h-3 bg-green-400 rounded-full shadow-[0_0_10px_#4ade80]"></div>
        </div>

        <!-- Chat Area -->
        <div id="chat-box" class="flex-1 p-6 overflow-y-auto space-y-6 bg-gray-50">
            <!-- Initial Bot Message -->
            <div class="flex gap-4">
                <div class="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center shrink-0">
                    <span class="text-blue-600 text-xs font-bold">AI</span>
                </div>
                <div class="bg-white p-4 rounded-2xl rounded-tl-none shadow-sm text-gray-700 text-sm border border-gray-100">
                    ¡Hola! Soy tu asistente de Aquaservice. ¿En qué puedo ayudarte hoy?
                </div>
            </div>
        </div>

        <!-- Input Area -->
        <div class="p-4 bg-white border-t border-gray-100">
            <form id="chat-form" class="relative group">
                <input type="text" id="user-input" 
                    class="w-full pl-6 pr-14 py-4 bg-gray-50 border border-gray-200 rounded-full text-gray-700 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all font-light"
                    placeholder="Escribe tu pregunta aquí..." autocomplete="off">
                <button type="submit" 
                    class="absolute right-2 top-2 p-2 bg-aquaservice text-white rounded-full hover:bg-blue-800 transition-colors shadow-md disabled:opacity-50 disabled:cursor-not-allowed">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14M12 5l7 7-7 7"/></svg>
                </button>
            </form>
            <p class="text-center text-xs text-gray-300 mt-2">Powered by RAG Engine • Gemini + Pinecone</p>
        </div>

    </div>

    <!-- Templates -->
    <template id="user-msg-template">
        <div class="flex gap-4 flex-row-reverse">
            <div class="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center shrink-0">
                <svg class="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/></svg>
            </div>
            <div class="bg-aquaservice p-4 rounded-2xl rounded-tr-none shadow-md text-white text-sm"></div>
        </div>
    </template>

    <template id="ai-msg-template">
        <div class="flex gap-4 group">
            <div class="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center shrink-0">
                <span class="text-blue-600 text-xs font-bold">AI</span>
            </div>
            <div class="relative max-w-[85%]">
                <div class="bg-white p-4 rounded-2xl rounded-tl-none shadow-sm text-gray-700 text-sm border border-gray-100 leading-relaxed answer-text"></div>
                
                <!-- Info Icon Wrapper -->
                <div class="absolute -right-12 top-0 md:group-hover:opacity-100 transition-opacity flex items-center h-full">
                   <div class="relative">
                        <button class="info-btn text-gray-300 hover:text-blue-500 transition-colors p-2">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                        </button>
                        <!-- Tooltip with Metadata -->
                        <div class="tooltip absolute bottom-full mb-2 -left-32 w-64 bg-slate-800 text-white text-xs rounded-lg p-3 shadow-xl z-50">
                            <p class="font-semibold mb-1 border-b border-gray-600 pb-1">Fuentes de conocimiento:</p>
                            <ul class="source-list space-y-1 text-gray-300"></ul>
                            <div class="absolute -bottom-1 left-1/2 -ml-1 w-2 h-2 bg-slate-800 transform rotate-45"></div>
                        </div>
                   </div>
                </div>
            </div>
        </div>
    </template>
    
    <template id="loading-template">
         <div class="flex gap-4" id="loading-dots">
            <div class="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center shrink-0">
                <span class="text-blue-600 text-xs font-bold">AI</span>
            </div>
            <div class="bg-white p-4 rounded-2xl rounded-tl-none shadow-sm border border-gray-100 flex items-center gap-1 h-12">
                <div class="w-2 h-2 bg-blue-400 rounded-full typing-dot"></div>
                <div class="w-2 h-2 bg-blue-400 rounded-full typing-dot"></div>
                <div class="w-2 h-2 bg-blue-400 rounded-full typing-dot"></div>
            </div>
        </div>
    </template>

    <script>
        const form = document.getElementById('chat-form');
        const input = document.getElementById('user-input');
        const chatBox = document.getElementById('chat-box');
        
        // Templates
        const userTpl = document.getElementById('user-msg-template');
        const aiTpl = document.getElementById('ai-msg-template');
        const loadTpl = document.getElementById('loading-template');

        function scrollToBottom() {
            chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: 'smooth' });
        }

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const message = input.value.trim();
            if (!message) return;

            // 1. Add User Message
            input.value = '';
            const userClone = userTpl.content.cloneNode(true);
            userClone.querySelector('div.text-white').textContent = message;
            chatBox.appendChild(userClone);
            scrollToBottom();

            // 2. Show Loading
            const loadingClone = loadTpl.content.cloneNode(true);
            chatBox.appendChild(loadingClone);
            scrollToBottom();
            const loadingEl = chatBox.lastElementChild;

            try {
                // 3. API Call
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                
                if (!res.ok) throw new Error("Server error");
                
                const data = await res.json();

                // 4. Remove Loading & Add AI Message
                loadingEl.remove();
                
                const aiClone = aiTpl.content.cloneNode(true);
                // Markdown-ish basic formatting (optional: use library for full md)
                aiClone.querySelector('.answer-text').innerText = data.answer; 
                
                // Populate Metadata
                const list = aiClone.querySelector('.source-list');
                if (data.sources && data.sources.length > 0) {
                    data.sources.forEach(src => {
                        const li = document.createElement('li');
                        li.textContent = `📄 ${src.file_name} (Score: ${src.score})`;
                        list.appendChild(li);
                    });
                } else {
                    const li = document.createElement('li');
                    li.textContent = "Sin fuentes directas.";
                    list.appendChild(li);
                }
                
                // Interactive Tooltip Logic
                const btn = aiClone.querySelector('.info-btn');
                const tooltip = aiClone.querySelector('.tooltip');
                
                // Click for mobile/desktop toggle
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const isVisible = tooltip.style.display === 'block';
                    document.querySelectorAll('.tooltip').forEach(t => t.style.display = 'none'); // close others
                    tooltip.style.display = isVisible ? 'none' : 'block';
                });
                
                // Close when clicking outside
                document.addEventListener('click', () => {
                    tooltip.style.display = 'none';
                });

                chatBox.appendChild(aiClone);
                scrollToBottom();

            } catch (err) {
                loadingEl.remove();
                console.error(err);
                const errDiv = document.createElement('div');
                errDiv.className = "text-center text-red-400 text-xs py-2";
                errDiv.textContent = "Error de conexión. Inténtalo de nuevo.";
                chatBox.appendChild(errDiv);
            }
        });
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
