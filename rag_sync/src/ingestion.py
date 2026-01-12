import logging
import google.generativeai as genai
from typing import List, Dict, Any
from .config import Config

logger = logging.getLogger("rag_sync.ingestion")

class Processor:
    def __init__(self, google_api_key: str):
        genai.configure(api_key=google_api_key)

    def chunk_text(self, text: str, chunk_size: int = 3000, chunk_overlap: int = 400) -> List[str]:
        """
        Simple character-based chunking.
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunks.append(text[start:end])
            
            if end == text_len:
                break
                
            start += (chunk_size - chunk_overlap)
            
        return chunks

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts using Google Gemini.
        """
        if not texts:
            return []
            
        try:
            # Gemini SDK handles batching but has limits per request (usually 100).
            # We'll batch it just in case our chunk lists are huge.
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                # embed_content can take a list for 'content'
                result = genai.embed_content(
                    model=Config.EMBEDDING_MODEL,
                    content=batch,
                    task_type="retrieval_document"
                )
                
                # result['embedding'] is a list of lists if input is a list
                if 'embedding' in result:
                    all_embeddings.extend(result['embedding'])
                    
            return all_embeddings

        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def process_file(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Takes raw file content and metadata, chunks it, embeds it,
        and returns a list of vector records ready for Pinecone.
        Format: { "id": str, "values": List[float], "metadata": Dict }
        """
        chunks = self.chunk_text(content, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        logger.info(f"Generated {len(chunks)} chunks for file {metadata.get('file_id')}")
        
        embeddings = self.embed_batch(chunks)
        
        vectors = []
        for i, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{metadata['file_id']}_{i}"
            
            # Enrich metadata
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = i
            chunk_metadata['text'] = chunk_text 
            
            vectors.append({
                "id": chunk_id,
                "values": vector,
                "metadata": chunk_metadata
            })
            
        return vectors
