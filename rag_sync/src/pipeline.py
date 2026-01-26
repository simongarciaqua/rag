import logging
import time
from typing import List, Dict, Any

from .config import Config
from .state import StateManager
from .drive_connector import DriveConnector
from .sf_connector import SalesforceConnector
from .ingestion import Processor
from .vector_store import VectorStore

logger = logging.getLogger("rag_sync.pipeline")

class SyncPipeline:
    def __init__(self):
        Config.validate()
        
        self.state_manager = StateManager(Config.STATE_FILE_PATH)
        self.drive = DriveConnector(Config.GOOGLE_SERVICE_ACCOUNT_FILE)
        self.sf = SalesforceConnector() # Nuevo conector SF
        self.processor = Processor(Config.GOOGLE_API_KEY)
        self.vector_store = VectorStore(
            Config.PINECONE_API_KEY, 
            Config.PINECONE_INDEX_NAME,
            Config.PINECONE_NAMESPACE
        )

    def run(self):
        logger.info("--- Iniciando Sincronización Híbrida (Drive + Salesforce) ---")
        
        # 1. Ejecutar Sincronización de Google Drive
        self.sync_google_drive()
        
        # 2. Ejecutar Sincronización de Salesforce Knowledge
        if Config.SF_USERNAME:
            self.sync_salesforce_knowledge()
        
        logger.info("--- Sincronización Finalizada con Éxito ---")

    def sync_google_drive(self):
        logger.info("Sincronizando Google Drive...")
        try:
            drive_files = self.drive.list_files_in_folder(Config.GOOGLE_DRIVE_FOLDER_ID)
            remote_files_map = {f['id']: f for f in drive_files if f.get('mimeType') != 'application/vnd.google-apps.folder'}
            
            local_file_ids = self.state_manager.get_all_file_ids()
            remote_file_ids = set(remote_files_map.keys())

            # Detectar cambios
            added = remote_file_ids - local_file_ids
            removed = local_file_ids - remote_file_ids
            
            # Procesar
            for fid in removed:
                self.vector_store.delete_by_file_id(fid)
                self.state_manager.remove_file(fid)
            
            for fid in added:
                f_meta = remote_files_map[fid]
                content = self.drive.download_file_content(fid, f_meta['mimeType'])
                if content:
                    meta = {"file_id": fid, "file_name": f_meta['name'], "source": "google_drive"}
                    vectors = self.processor.process_content(content, meta)
                    if vectors:
                        self.vector_store.upsert(vectors)
                        self.state_manager.update_file(fid, f_meta['modifiedTime'])

        except Exception as e:
            logger.error(f"Error en sincronización Drive: {e}")

    def sync_salesforce_knowledge(self):
        logger.info("Sincronizando Salesforce Knowledge...")
        try:
            articles = self.sf.get_knowledge_articles()
            
            for art in articles:
                art_id = art['Id']
                remote_mod = art['LastModifiedDate']
                local_mod = self.state_manager.get_modified_time(art_id)

                if remote_mod != local_mod:
                    logger.info(f"Actualizando artículo SF: {art['Title']}")
                    
                    # Obtener contenido completo
                    details = self.sf.get_article_details(art_id)
                    if details:
                        # Eliminar versiones viejas en Pinecone
                        self.vector_store.delete_by_file_id(art_id)
                        
                        # Procesar nuevo contenido (HTML -> Text -> Chunks -> Embeddings)
                        meta = {
                            "file_id": art_id,
                            "file_name": details['title'],
                            "source": "salesforce_knowledge",
                            "url": f"{Config.SF_DOMAIN}.lightning.force.com/{art_id}"
                        }
                        
                        vectors = self.processor.process_content(details['html'], meta, is_html=True)
                        if vectors:
                            self.vector_store.upsert(vectors)
                            self.state_manager.update_file(art_id, remote_mod)
        
        except Exception as e:
            logger.error(f"Error en sincronización Salesforce: {e}")
