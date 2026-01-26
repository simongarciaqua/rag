import logging
import uuid
from typing import List, Dict, Any
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .config import Config

logger = logging.getLogger("rag_sync.ingestion")

class Processor:
    def __init__(self, api_key: str):
        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )
        # LangChain Chunker: Más inteligente que un split simple
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def clean_html(self, html_content: str) -> str:
        """Convierte HTML de Salesforce Knowledge en texto plano limpio."""
        if not html_content:
            return ""
        soup = BeautifulSoup(html_content, "html.parser")
        # Eliminar scripts y estilos
        for script in soup(["script", "style"]):
            script.extract()
        
        # Obtener texto con saltos de línea coherentes
        text = soup.get_text(separator='\n')
        
        # Limpiar espacios en blanco extra
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text

    def process_content(self, text_content: str, metadata: Dict[str, Any], is_html: bool = False) -> List[Dict[str, Any]]:
        """Limpia, trocea y genera embeddings para cualquier contenido."""
        
        # 1. Limpieza si es necesario
        if is_html:
            text_content = self.clean_html(text_content)
        
        if not text_content or len(text_content) < 10:
            return []

        # 2. Chunking con LangChain
        chunks = self.text_splitter.split_text(text_content)
        logger.info(f"Contenido dividido en {len(chunks)} chunks.")

        # 3. Generar Embeddings masivos
        try:
            # LangChain maneja el batching automáticamente
            embeddings = self.embeddings_model.embed_documents(chunks)
        except Exception as e:
            logger.error(f"Error generando embeddings: {e}")
            return []

        # 4. Formatear para Pinecone
        vectors = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{metadata['file_id']}_{i}"
            
            # Combinamos metadata original con el texto del chunk
            chunk_metadata = metadata.copy()
            chunk_metadata["text"] = chunk_text
            chunk_metadata["chunk_index"] = i
            
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": chunk_metadata
            })
            
        return vectors
