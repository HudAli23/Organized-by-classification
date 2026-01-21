"""Main pipeline: validate paths -> collect files -> extract text -> classify -> build folder structure -> move files."""
import os
import logging
from core.scanner import collect_all_files
from classification.classifier import classify_document
from core.organizer import move_file, safe_delete_directory
from utils.extraction import extract_text_from_file
from semantic.organization import get_organization_structure


def validate_paths(src, dest):
    """Validate source and destination paths.
    
    Args:
        src: Source directory path
        dest: Destination directory path
        
    Returns:
        Tuple of (valid: bool, error_message: str)
    """
    # Check source
    if not src or not isinstance(src, str):
        return False, "Invalid source path"
    
    if not os.path.isdir(src):
        return False, f"Source directory not found: {src}"
    
    if not os.access(src, os.R_OK):
        return False, f"No read permission for source: {src}"
    
    # Check destination
    if not dest or not isinstance(dest, str):
        return False, "Invalid destination path"
    
    if src == dest:
        return False, "Source and destination cannot be the same"
    
    # Check if destination is subdirectory of source
    src_abs = os.path.abspath(src)
    dest_abs = os.path.abspath(dest)
    
    if dest_abs.startswith(src_abs):
        logging.warning("Destination is inside source directory")
    
    return True, ""


def run_pipeline(src, dest, progress_callback=None, use_dynamic_folders=True):
    """Run the file organization pipeline.
    
    Args:
        src: Source directory to organize
        dest: Destination directory for organized files
        progress_callback: Optional callback function(current, total, message)
        use_dynamic_folders: If True, create folders based on content relationships
        
    Returns:
        Dict with statistics: {'processed': int, 'moved': int, 'failed': int, 'structure': dict}
    """
    # Validate paths
    valid, error = validate_paths(src, dest)
    if not valid:
        logging.error(error)
        return {'processed': 0, 'moved': 0, 'failed': 0, 'structure': {}}
    
    # Collect files
    files = collect_all_files(src)
    if not files:
        logging.info("No files found in source directory")
        return {'processed': 0, 'moved': 0, 'failed': 0, 'structure': {}}
    
    logging.info("Found %d files to process", len(files))
    
    # First pass: Extract text and classify all files
    documents_info = []
    file_cache = {}
    
    for idx, file_path in enumerate(files):
        if progress_callback:
            progress_callback(idx + 1, len(files), f"Analyzing: {os.path.basename(file_path)}")
        
        try:
            text = extract_text_from_file(file_path)
            label, confidence = classify_document(text, file_path)
            
            documents_info.append({
                'path': file_path,
                'filename': os.path.basename(file_path),
                'text': text,
                'theme': label,
                'confidence': confidence
            })
            
            file_cache[file_path] = label
            
        except Exception as e:
            logging.error("Error analyzing file %s: %s", file_path, e)
            file_cache[file_path] = "Others"
    
    # Second pass: Generate organization structure
    if use_dynamic_folders and len(documents_info) > 1:
        logging.info("Generating dynamic folder structure based on relationships...")
        try:
            organization_structure = get_organization_structure(documents_info)
        except Exception as e:
            logging.warning("Dynamic organization failed: %s, using themes", e)
            # Fallback to theme-based organization
            organization_structure = {}
            for doc_info in documents_info:
                theme = doc_info['theme']
                if theme not in organization_structure:
                    organization_structure[theme] = []
                organization_structure[theme].append(doc_info['path'])
    else:
        # Use theme-based folders
        organization_structure = {}
        for file_path, theme in file_cache.items():
            if theme not in organization_structure:
                organization_structure[theme] = []
            organization_structure[theme].append(file_path)
    
    # Third pass: Move files to organized folders
    stats = {'processed': len(files), 'moved': 0, 'failed': 0, 'structure': organization_structure}
    
    for folder_name, file_paths in organization_structure.items():
        dest_dir = os.path.join(dest, folder_name)
        
        for file_path in file_paths:
            try:
                if progress_callback:
                    progress_callback(
                        stats['moved'] + stats['failed'],
                        len(files),
                        f"Moving: {os.path.basename(file_path)} -> {folder_name}"
                    )
                
                move_file(file_path, dest_dir)
                stats['moved'] += 1
                
                logging.info(
                    "Moved: %s -> %s",
                    os.path.basename(file_path),
                    folder_name
                )
                
            except Exception as e:
                logging.error("Error moving file %s: %s", file_path, e)
                stats['failed'] += 1
    
    # Clean up empty source subdirectories
    try:
        for root, dirs, files_in_dir in os.walk(src, topdown=False):
            if not files_in_dir and root != src:
                safe_delete_directory(root)
    except Exception as e:
        logging.warning("Error cleaning up empty directories: %s", e)
    
    logging.info(
        "Pipeline complete: %d processed, %d moved, %d failed",
        stats['processed'],
        stats['moved'],
        stats['failed']
    )
    
    logging.info("Folder structure created:")
    for folder_name, file_list in organization_structure.items():
        logging.info("  %s/ (%d files)", folder_name, len(file_list))
    
    return stats

