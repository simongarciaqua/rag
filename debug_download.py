import logging
import sys
import os

# Add rag_sync to pythonpath to allow src imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'rag_sync'))

from src.config import Config
from src.drive_connector import DriveConnector

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_download")

# ID from previous logs for Aquamigo.docx
FILE_ID = "1eNjR9A1vZmjMpsNVSO9QXYBOPE4KQCPW" 
MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

def debug_run():
    Config.validate()
    drive = DriveConnector(Config.GOOGLE_SERVICE_ACCOUNT_FILE)
    
    logger.info(f"Downloading file {FILE_ID}...")
    content = drive.download_file_content(FILE_ID, MIME_TYPE)
    
    if content:
        print("\n--- EXTRACTED CONTENT START ---")
        print(content)
        print("--- EXTRACTED CONTENT END ---\n")
        
        if "aquamigo seguro" in content.lower():
            print("✅ FOUND 'aquamigo seguro' in downloaded content.")
        else:
            print("❌ DID NOT FIND 'aquamigo seguro' in downloaded content.")
    else:
        print("❌ No content extracted.")

if __name__ == "__main__":
    debug_run()
