import os
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone
from api.rule_engine import RuleEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Helpers
def clean_key(val):
    if not val: return ""
    if '=' in val: val = val.split('=')[-1]
    import re
    return re.sub(r'[\s\n\r\t]', '', val).strip("'\" ")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP_RULES_PATH = os.path.join(BASE_DIR, "stop_reparto", "rules_stop_reparto.json")
URGENT_RULES_PATH = os.path.join(BASE_DIR, "aviso_urgente", "rules_aviso_urgente.json")

class AgentSystem:
    def __init__(self):
        self._setup_apis()
        self.stop_engine = RuleEngine(STOP_RULES_PATH)
        self.urgent_engine = RuleEngine(URGENT_RULES_PATH)
        
        # System Instructions for extraction
        self.extraction_model = genai.GenerativeModel('models/gemini-2.0-flash')
        self.chat_model = genai.GenerativeModel('models/gemini-2.0-flash', system_instruction="Eres un asistente virtual de Aquaservice. Tu misión es ayudar al cliente con sus pedidos, dudas y gestiones. Responde SIEMPRE en español de forma amable y profesional.")
        
    def _setup_apis(self):
        google_key = clean_key(os.getenv('GOOGLE_API_KEY'))
        pc_key = clean_key(os.getenv('PINECONE_API_KEY'))
        self.index_name = clean_key(os.getenv('PINECONE_INDEX_NAME'))
        self.pc_namespace = clean_key(os.getenv('PINECONE_NAMESPACE', 'default'))

        genai.configure(api_key=google_key)
        self.pc = Pinecone(api_key=pc_key)
        self.index = self.pc.Index(self.index_name)

    async def process_request(self, message: str, history: list, context: dict, flow_step: str = None):
        """
        Main entry point.
        1. Analyze Turn (Is it an answer to current flow? or a new intent?)
        2. Handle Interruption vs Flow Continuation.
        3. Route to specific Agent logic.
        """
        
        # 1. Analyze the turn structure
        analysis = await self._analyze_turn(message, history, flow_step)
        logger.info(f"Turn Analysis: {analysis}")
        
        intent = analysis.get("intent")
        is_interruption = analysis.get("is_interruption", False)
        
        response = {
            "answer": "",
            "context_updates": {},
            "flow_step": None,
            "source": "CHAT"
        }

        # 2. Logic for Interruption (e.g. User asks FAQ while in STOP flow)
        if is_interruption:
            # Execute the interrupting intent
            if intent == "FAQ":
                # We handle the FAQ normally
                sub_response = await self._handle_rag(message, history)
                
                # We append a "Resumption Hint" to the answer
                # And we RETURN the ORIGINAL flow_step so frontend stays in that state
                response["answer"] = sub_response["answer"] + "\n\n_(Por cierto, seguimos pendientes de tu gestión anterior. ¿Deseas continuar?)_"
                response["source"] = "RAG" # Show RAG source
                response["sources_data"] = sub_response["sources_data"]
                response["flow_step"] = flow_step # CRITICAL: Resuming original flow
                
                return response
            
            elif intent in ["STOP_DELIVERY", "URGENT_NOTICE"]:
                # Full Context Switch
                # If they switch, we just let the Routing logic handle the new intent.
                # flow_step will update to the new one.
                pass 
        
        # If answering flow, force intent to be the flow type
        if intent == "ANSWER_FLOW" and flow_step:
            intent = flow_step.split('_')[0] + "_" + flow_step.split('_')[1] # Approximation, or just use flow_step prefixes
            if "STOP" in flow_step: intent = "STOP_DELIVERY"
            if "URGENT" in flow_step: intent = "URGENT_NOTICE"


        # 3. Routing
        if intent == "STOP_DELIVERY" or (flow_step and flow_step.startswith("STOP_") and not is_interruption):
            response = await self._handle_stop_delivery(message, context)
        elif intent == "URGENT_NOTICE" or (flow_step and flow_step.startswith("URGENT_") and not is_interruption):
            response = await self._handle_urgent_notice(message, context)
        elif intent == "FAQ":
            response = await self._handle_rag(message, history)
        else:
            # Fallback / Chat
            response["answer"] = await self._simple_chat(message, history)
            response["source"] = "CHAT"

        return response

    async def _analyze_turn(self, message, history, current_flow_step):
        """
        Decides if the message is:
        - ANSWER_FLOW: Answering the active flow's question.
        - INTERRUPTION: Asking something unrelated (FAQ).
        - NEW_INTENT: Explicitly changing process.
        """
        if not current_flow_step:
            # If no flow, just classify intent standardly
            intent = await self._classify_intent(message, history)
            return {"intent": intent, "is_interruption": False}

        # If we are in a flow, we need to be smarter
        prompt = f"""
        System is currently in a flow state: '{current_flow_step}'.
        User just sent: "{message}"
        
        Analyze the user's message in relation to the flow.
        1. ANSWER_FLOW: Is the user answering the flow's question OR accepting one of the offered alternatives? (e.g. "Yes", "Change it for minis", "I want option 2", "exceso de agua").
        2. INTERRUPTION: Is the user asking a generic knowledge question unrelated to completing the current action? (e.g. "What is the pH?", "Where are you located?") -> Intent: FAQ
        3. NEW_INTENT: Is the user explicitly abandoning this and starting a different process? (e.g. "Forget it", "I want an urgent delivery instead") -> Intent: STOP/URGENT
        
        Return JSON ONLY:
        {{
            "classification": "ANSWER_FLOW" | "INTERRUPTION" | "NEW_INTENT",
            "detected_intent": "FAQ" | "STOP_DELIVERY" | "URGENT_NOTICE" | "CHAT"
        }}
        """
        try:
            res = self.chat_model.generate_content(prompt)
            text = res.text.replace('```json', '').replace('```', '').strip()
            data = json.loads(text)
            
            if data["classification"] == "ANSWER_FLOW":
                return {"intent": "ANSWER_FLOW", "is_interruption": False}
            elif data["classification"] == "INTERRUPTION":
                return {"intent": data["detected_intent"], "is_interruption": True}
            else:
                return {"intent": data["detected_intent"], "is_interruption": False} # Context Switch
        except Exception as e:
            logger.error(f"Error in analyze_turn: {e}")
            # Fallback: Check keywords manually
            msg_lower = message.lower()
            if "cambia" in msg_lower or "botella" in msg_lower or "caja" in msg_lower or "si" in msg_lower or "no" in msg_lower:
                 return {"intent": "ANSWER_FLOW", "is_interruption": False}
            
            intent = await self._classify_intent(message, history)
            return {"intent": intent, "is_interruption": intent != "CHAT"}


    async def _classify_intent(self, message, history):
        hist_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-2:]])
        prompt = f"""
        Analyze the user's latest message and conversation history.
        Classify the intent into one of these exact categories:
        - STOP_DELIVERY: User wants to stop, pause, cancel, or skip a delivery/service.
        - URGENT_NOTICE: User needs water immediately, ran out, urgent delivery request.
        - FAQ: User asks a general question about service, prices, bottles, machines, etc.
        - CHAT: Greetings, small talk, or unclear.

        History:
        {hist_text}

        User Message: {message}

        Category:
        """
        res = self.chat_model.generate_content(prompt)
        text = res.text.strip().upper()
        if "STOP" in text: return "STOP_DELIVERY"
        if "URGENT" in text: return "URGENT_NOTICE"
        if "FAQ" in text: return "FAQ"
        return "CHAT"

    async def _handle_stop_delivery(self, message, context):
        # 1. Check Missing Info
        missing = self.stop_engine.get_missing_info(context)
        
        if missing:
            # We need to extract if the user just provided it
            extracted = await self._extract_field(message, missing['missing_field'], missing.get('options'))
            if extracted:
                context[missing['missing_field']] = extracted
                missing = self.stop_engine.get_missing_info(context)
                if not missing:
                    pass
                else:
                    return {
                        "answer": f"Gracias. " + missing['question'],
                        "context_updates": {missing['missing_field']: extracted} if extracted else {},
                        "flow_step": "STOP_DELIVERY",
                        "source": "AGENT_STOP",
                        "debug_info": {"step": "Collecting Info", "details": f"Missing: {missing['missing_field']}"}
                    }
            
            if missing:
                 return {
                    "answer": missing['question'],
                    "context_updates": context,
                    "flow_step": "STOP_DELIVERY",
                    "source": "AGENT_STOP",
                    "debug_info": {"step": "Collecting Info", "details": f"Missing: {missing['missing_field']}"}
                }

        # 2. Evaluate Rules
        decision = self.stop_engine.evaluate(context)
        
        if decision:
             answer = await self._generate_decision_response(decision, "STOP_DELIVERY", context)
             return {
                 "answer": answer,
                 "context_updates": context,
                 "flow_step": None, 
                 "source": "AGENT_STOP",
                 "debug_info": decision
             }
        
        return {
            "answer": "No he podido determinar una acción automática para tu caso. Un agente humano revisará tu solicitud.",
            "context_updates": {},
            "flow_step": None,
            "source": "AGENT_STOP",
            "debug_info": {"error": "No rule matched"}
        }

    async def _handle_urgent_notice(self, message, context):
        # 1. Check Missing Info
        missing = self.urgent_engine.get_missing_info(context)
        
        if missing:
            extracted = await self._extract_field(message, missing['missing_field'], missing.get('options'))
            
            # Logic for 'producto' inference
            if missing['missing_field'] == 'producto' and not extracted:
                lower = message.lower()
                if "agua" in lower: extracted = "agua"
                elif "cafe" in lower or "café" in lower: extracted = "cafe"
                
            if extracted:
                context[missing['missing_field']] = extracted
                missing = self.urgent_engine.get_missing_info(context)
                if not missing:
                    pass
                else:
                    return {
                        "answer": missing['question'],
                        "context_updates": {missing['missing_field']: extracted} if extracted else {},
                        "flow_step": "URGENT_NOTICE",
                        "source": "AGENT_URGENT",
                        "debug_info": {"step": "Collecting Info", "details": f"Missing: {missing['missing_field']}"}
                    }
            else:
                 return {
                    "answer": missing['question'],
                    "context_updates": {},
                    "flow_step": "URGENT_NOTICE",
                    "source": "AGENT_URGENT",
                    "debug_info": {"step": "Collecting Info", "details": f"Missing: {missing['missing_field']}"}
                }

        # 2. Evaluate Rules
        decision = self.urgent_engine.evaluate(context)
        if decision:
            answer = await self._generate_decision_response(decision, "URGENT_NOTICE", context)
            return {
                "answer": answer,
                "context_updates": context,
                "flow_step": None,
                "source": "AGENT_URGENT",
                "debug_info": decision
            }
        
        return {
             "answer": "No podemos procesar el aviso urgente con los datos actuales.",
             "flow_step": None,
             "source": "AGENT_URGENT",
             "context_updates": {},
             "debug_info": {"error": "No rule matched"}
        }

    async def _handle_rag(self, message, history):
        embed = genai.embed_content(model='models/text-embedding-004', content=message, task_type="retrieval_query")
        results = self.index.query(vector=embed['embedding'], top_k=3, include_metadata=True, namespace=self.pc_namespace)
        matches = [m for m in results.matches if m.score > 0.45]
        context_text = "\n".join([m.metadata.get('text', '') for m in matches])
        
        prompt = f"""
        Responde a la pregunta del usuario basándote ÚNICAMENTE en el siguiente contexto.
        Responde en Español de forma clara y concisa.
        Contexto:
        {context_text}
        
        Pregunta Usuario: {message}
        """
        res = self.chat_model.generate_content(prompt)
        sources = [{"name": m.metadata.get('file_name', 'Doc'), "score": round(m.score*100, 1)} for m in matches]
        
        return {
            "answer": res.text,
            "context_updates": {},
            "flow_step": None,
            "source": "RAG",
            "sources_data": sources,
            "debug_info": {"context_used": [s['name'] for s in sources]}
        }
    async def _simple_chat(self, message, history):
        # Basic chat
        gemini_history = [{"role": "user" if m['role'] == "user" else "model", "parts": [m['content']]} for m in history]
        chat = self.chat_model.start_chat(history=gemini_history)
        res = chat.send_message(message)
        return res.text

    async def _extract_field(self, message, field_name, options):
        # Logic to extract value for a field from natural language
        opt_str = ""
        if options:
            opt_str = "Possible values: " + ", ".join([f"{o['value']} ({o['label']})" for o in options])
        
        prompt = f"""
        Extract the value for the field '{field_name}' from the text: "{message}".
        {opt_str}
        If the text contains the answer, return ONLY the value (e.g. from the 'value' field if options are provided).
        If not found, return NULL.
        """
        res = self.chat_model.generate_content(prompt)
        val = res.text.strip()
        if "NULL" in val or not val: return None
        # Basic cleanup
        return val.replace('"', '').replace("'", "")

    async def _generate_decision_response(self, decision, process_type, context):
        allowed_actions = decision.get('allowed_actions', [])
        reason = decision.get('reason', '')
        desc = decision.get('decision', '')
        
        # Select Policy Text
        policy_text = ""
        try:
            if "STOP" in process_type:
                with open(STOP_RULES_PATH.replace('rules_stop_reparto.json', 'policy_stop_reparto.txt'), 'r') as f:
                    policy_text = f.read()
            elif "URGENT" in process_type:
                with open(URGENT_RULES_PATH.replace('rules_aviso_urgente.json', 'policy_aviso_urgente.txt'), 'r') as f:
                    policy_text = f.read()
        except Exception as e:
            logger.error(f"Error loading policy text: {e}")

        
        prompt = f"""
        Genera una respuesta FINAL para el cliente.
        NO expliques qué regla has usado.
        NO enumeres parámetros técnicos.
        Se DIRECTO, AMABLE y BREVE.
        
        Si la decisión es APROBADA: Confirma la acción y da el plazo de entrega.
        Si es RECHAZADA: Explica el motivo de forma sencilla (ej: "Por tu zona no podemos...") y ofrece la alternativa si la hay.
        
        MANUAL OPERATIVO (Solo para extraer datos como plazos):
        {policy_text}
        
        SITUACIÓN:
        Proceso: {process_type}
        Decisión: {desc}
        Motivo Técnico: {reason}
        Acciones: {allowed_actions}
        Usuario Contexto: {json.dumps(context)}
        """
        res = self.chat_model.generate_content(prompt)
        return res.text

