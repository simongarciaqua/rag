import logging
import time
from typing import List, Dict, Any

from .config import Config
from .state import StateManager
from .drive_connector import DriveConnector
from .ingestion import Processor
from .vector_store import VectorStore

logger = logging.getLogger("rag_sync.pipeline")

class SyncPipeline:
    def __init__(self):
        Config.validate()
        
        self.state_manager = StateManager(Config.STATE_FILE_PATH)
        self.drive = DriveConnector(Config.GOOGLE_SERVICE_ACCOUNT_FILE)
        self.processor = Processor(Config.GOOGLE_API_KEY)
        self.vector_store = VectorStore(
            Config.PINECONE_API_KEY, 
            Config.PINECONE_INDEX_NAME,
            Config.PINECONE_NAMESPACE
        )

    def run(self):
        logger.info("Starting Sync Job...")
        
        # 1. Fetch Remote State
        try:
            drive_files = self.drive.list_files_in_folder(Config.GOOGLE_DRIVE_FOLDER_ID)
            # Map id -> file_obj
            remote_files_map = {f['id']: f for f in drive_files if f.get('mimeType') != 'application/vnd.google-apps.folder'}
        except Exception as e:
            logger.critical(f"Failed to fetch Drive files: {e}")
            return

        # 2. Get Local State
        local_file_ids = self.state_manager.get_all_file_ids()
        remote_file_ids = set(remote_files_map.keys())

        # 3. Compute Diff
        added_file_ids = remote_file_ids - local_file_ids
        removed_file_ids = local_file_ids - remote_file_ids
        
        updated_file_ids = set()
        for fid in (remote_file_ids & local_file_ids):
            remote_mod = remote_files_map[fid]['modifiedTime']
            local_mod = self.state_manager.get_modified_time(fid)
            if remote_mod != local_mod:
                updated_file_ids.add(fid)

        logger.info(f"Sync Plan: {len(added_file_ids)} to add, {len(updated_file_ids)} to update, {len(removed_file_ids)} to remove.")

        # 4. Handle Deletions
        for fid in removed_file_ids:
            self.handle_deletion(fid)

        # 5. Handle Additions & Updates
        # Treat updates as delete + add to ensure no stale chunks remain
        for fid in updated_file_ids:
            # Delete old chunks first
            self.vector_store.delete_by_file_id(fid)
            self.handle_ingestion(remote_files_map[fid])

        for fid in added_file_ids:
            self.handle_ingestion(remote_files_map[fid])

        logger.info("Sync Job Completed.")

    def handle_deletion(self, file_id: str):
        logger.info(f"Processing Deletion: {file_id}")
        try:
            self.vector_store.delete_by_file_id(file_id)
            self.state_manager.remove_file(file_id)
        except Exception as e:
            logger.error(f"Failed to process deletion for {file_id}: {e}")

    def handle_ingestion(self, file_meta: Dict[str, Any]):
        file_id = file_meta['id']
        file_name = file_meta['name']
        modified_time = file_meta['modifiedTime']
        mime_type = file_meta['mimeType']

        logger.info(f"Processing Ingestion: {file_name} ({file_id})")

        try:
            # A. Download
            content = self.drive.download_file_content(file_id, mime_type)
            if not content:
                logger.warning(f"No content extracted for {file_id}, skipping.")
                return

            # B. Prepare Metadata
            metadata = {
                "file_id": file_id,
                "file_name": file_name,
                "source": "google_drive",
                "modified_time": modified_time,
                "synced_at": time.time()
            }

            # C. Chunk & Embed
            vectors = self.processor.process_file(content, metadata)
            
            # D. Upsert
            if vectors:
                self.vector_store.upsert(vectors)
                # E. Update State
                self.state_manager.update_file(file_id, modified_time)
            else:
                logger.warning(f"No vectors generated for {file_id}")

        except Exception as e:
            logger.error(f"Failed to ingest file {file_id}: {e}")

if __name__ == "__main__":
    pipeline = SyncPipeline()
    pipeline.run()
