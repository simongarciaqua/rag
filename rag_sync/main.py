import logging
import sys
import os

# Add src to pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import SyncPipeline

if __name__ == "__main__":
    try:
        pipeline = SyncPipeline()
        pipeline.run()
    except Exception as e:
        logging.critical(f"Unhandled exception in main execution: {e}")
        sys.exit(1)
