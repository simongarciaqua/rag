import io
import logging
from typing import List, Dict, Any, Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Config

logger = logging.getLogger("rag_sync.drive")

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class DriveConnector:
    def __init__(self, service_account_file: str):
        try:
            self.creds = service_account.Credentials.from_service_account_file(
                service_account_file, scopes=SCOPES
            )
            self.service = build('drive', 'v3', credentials=self.creds)
        except Exception as e:
            logger.error(f"Failed to initialize Drive client: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def list_files_in_folder(self, folder_id: str) -> List[Dict[str, Any]]:
        """
        List all files in the specific folder.
        Returns a list of dicts with id, name, modifiedTime, mimeType.
        """
        files = []
        page_token = None
        
        # Query: inside folder folder_id AND not trashed
        query = f"'{folder_id}' in parents and trashed = false"

        while True:
            try:
                response = self.service.files().list(
                    q=query,
                    spaces='drive',
                    fields='nextPageToken, files(id, name, modifiedTime, mimeType)',
                    pageToken=page_token
                ).execute()
                
                found_files = response.get('files', [])
                files.extend(found_files)
                
                page_token = response.get('nextPageToken', None)
                if page_token is None:
                    break
            except Exception as e:
                logger.error(f"Error listing files: {e}")
                raise

        return files

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def download_file_content(self, file_id: str, mime_type: str) -> Optional[str]:
        """
        Download content of a file. 
        Supports:
        - Google Docs (exports to text/plain)
        - Text files (text/*)
        - Word Docs (application/vnd.openxmlformats-officedocument.wordprocessingml.document)
        - PDF (application/pdf)
        """
        try:
            content_bytes = None
            
            # 1. Google Native Docs -> Export
            if mime_type == 'application/vnd.google-apps.document':
                request = self.service.files().export_media(fileId=file_id, mimeType='text/plain')
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    _, done = downloader.next_chunk()
                return fh.getvalue().decode('utf-8')

            # 2. Binary Downloads (Word, PDF, Text)
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                _, done = downloader.next_chunk()
            content_bytes = fh.getvalue()

            # 3. Parse based on Type
            if mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                try:
                    import docx
                    doc = docx.Document(io.BytesIO(content_bytes))
                    full_text = []
                    for para in doc.paragraphs:
                        full_text.append(para.text)
                    return '\n'.join(full_text)
                except ImportError:
                    logger.error("python-docx not installed, cannot read .docx")
                    return None
                except Exception as e:
                    logger.error(f"Error parsing DOCX {file_id}: {e}")
                    return None

            elif mime_type == 'application/pdf':
                try:
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(content_bytes))
                    full_text = []
                    for page in reader.pages:
                        full_text.append(page.extract_text() or "")
                    return '\n'.join(full_text)
                except ImportError:
                    logger.error("pypdf not installed, cannot read .pdf")
                    return None
                except Exception as e:
                    logger.error(f"Error parsing PDF {file_id}: {e}")
                    return None
            
            # 4. Fallback: Try decoding as UTF-8 text
            try:
                return content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                logger.warning(f"File {file_id} ({mime_type}) is binary and not a supported format (PDF/DOCX). Skipping.")
                return None

        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {e}")
            return None
