import logging
import time
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Config

logger = logging.getLogger("rag_sync.vector_store")

class VectorStore:
    def __init__(self, api_key: str, index_name: str, namespace: str):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.namespace = namespace
        
        # Ensure index exists (optional check, usually assumed created via IaC)
        # We'll just connect.
        try:
            self.index = self.pc.Index(index_name)
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone index {index_name}: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def upsert(self, vectors: List[Dict[str, Any]]):
        """
        Batch upsert vectors.
        """
        if not vectors:
            return

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            try:
                self.index.upsert(vectors=batch, namespace=self.namespace)
                logger.debug(f"Upserted batch {i} to {i+len(batch)}")
            except Exception as e:
                logger.error(f"Failed to upsert batch: {e}")
                raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def delete_by_file_id(self, file_id: str):
        """
        Delete all vectors associated with a file_id.
        Using metadata filtering.
        """
        try:
            self.index.delete(
                filter={"file_id": file_id}, 
                namespace=self.namespace
            )
            logger.info(f"Deleted vectors for file_id: {file_id}")
        except Exception as e:
            logger.error(f"Failed to delete vectors for {file_id}: {e}")
            raise 
