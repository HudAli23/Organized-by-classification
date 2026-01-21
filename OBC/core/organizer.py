"""File organization and movement utilities."""
import os
import shutil
import logging
from pathlib import Path


def move_file(src, dest_dir):
    """Move file to destination directory with conflict handling.
    
    Args:
        src: Source file path
        dest_dir: Destination directory path
        
    Raises:
        ValueError: If source file doesn't exist
        OSError: If operation fails after retries
    """
    if not os.path.exists(src):
        raise ValueError(f"Source file not found: {src}")
    
    if not os.path.isfile(src):
        raise ValueError(f"Source is not a file: {src}")
    
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except PermissionError:
        raise OSError(f"Permission denied creating directory: {dest_dir}")
    except Exception as e:
        raise OSError(f"Failed to create directory {dest_dir}: {e}")
    
    filename = os.path.basename(src)
    dest = os.path.join(dest_dir, filename)
    
    # Handle file conflicts
    if os.path.exists(dest):
        dest = _get_unique_filename(dest)
        logging.info("File exists, using unique name: %s", dest)
    
    try:
        shutil.move(src, dest)
        logging.info("Moved: %s -> %s", src, dest)
    except PermissionError:
        raise OSError(f"Permission denied moving file: {src}")
    except Exception as e:
        raise OSError(f"Failed to move file {src} to {dest}: {e}")


def _get_unique_filename(filepath):
    """Generate unique filename if file already exists.
    
    Args:
        filepath: Original file path
        
    Returns:
        Unique file path with number suffix
    """
    path = Path(filepath)
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    
    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            return str(new_path)
        counter += 1


def safe_delete_directory(directory):
    """Safely delete empty directories.
    
    Args:
        directory: Path to directory to delete
        
    Returns:
        True if deleted, False otherwise
    """
    try:
        if os.path.isdir(directory) and not os.listdir(directory):
            os.rmdir(directory)
            logging.info("Removed empty directory: %s", directory)
            return True
    except Exception as e:
        logging.warning("Failed to remove directory %s: %s", directory, e)
    return False
