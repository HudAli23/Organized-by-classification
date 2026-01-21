import os
import logging

def clean_empty_folders(root_dir: str):
    """
    Recursively remove empty directories.
    """
    for root, dirs, _ in os.walk(root_dir, topdown=False):
        for d in dirs:
            path = os.path.join(root, d)
            try:
                if not os.listdir(path):
                    os.rmdir(path)
                    logging.info(f"Removed empty folder: {path}")
            except Exception as e:
                logging.warning(f"Could not remove folder {path}: {e}")
