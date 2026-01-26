import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logger Config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_sync")

class Config:
    # Google Drive
    GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "credentials.json")
    GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

    # Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")  # Good for multitenancy

    # Gemini
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    EMBEDDING_MODEL = "models/text-embedding-004"
    
    # Processing
    # Gemini 004 window is huge, but for RAG chunks we still want manageable sizes.
    # ~3000 chars is roughly 600-800 tokens.
    CHUNK_SIZE = 3000
    CHUNK_OVERLAP = 400
    
    # State
    STATE_FILE_PATH = os.getenv("STATE_FILE_PATH", "sync_state.json")

    # Salesforce
    SF_USERNAME = os.getenv("SF_USERNAME")
    SF_PASSWORD = os.getenv("SF_PASSWORD")
    SF_SECURITY_TOKEN = os.getenv("SF_SECURITY_TOKEN")
    SF_CLIENT_ID = os.getenv("SF_CLIENT_ID")
    SF_CLIENT_SECRET = os.getenv("SF_CLIENT_SECRET")
    SF_DOMAIN = os.getenv("SF_DOMAIN", "login") # 'test' para sandbox

    @classmethod
    def validate(cls):
        required = [
            "PINECONE_API_KEY", "PINECONE_INDEX_NAME", 
            "GOOGLE_API_KEY", "GOOGLE_DRIVE_FOLDER_ID"
        ]
        # Validamos SF solo si se pretende usar
        if cls.SF_USERNAME:
            required.extend(["SF_PASSWORD", "SF_SECURITY_TOKEN"])
            
        missing = [key for key in required if not getattr(cls, key) and not os.getenv(key)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
