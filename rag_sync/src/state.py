import json
import os
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger("rag_sync.state")

class StateManager:
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.state: Dict[str, str] = {}  # file_id -> modified_time
        self.load()

    def load(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.state = data.get("files", {})
                logger.info(f"Loaded state with {len(self.state)} files.")
            except Exception as e:
                logger.error(f"Failed to load state file: {e}")
                self.state = {}
        else:
            logger.info("No existing state file found. Starting fresh.")
            self.state = {}

    def save(self):
        try:
            data = {
                "last_run": datetime.utcnow().isoformat(),
                "files": self.state
            }
            # Atomic write pattern to avoid corruption
            temp_file = self.state_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(temp_file, self.state_file)
            logger.debug("State saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def get_modified_time(self, file_id: str) -> Optional[str]:
        return self.state.get(file_id)

    def update_file(self, file_id: str, modified_time: str):
        self.state[file_id] = modified_time
        self.save()

    def remove_file(self, file_id: str):
        if file_id in self.state:
            del self.state[file_id]
            self.save()

    def get_all_file_ids(self):
        return set(self.state.keys())
