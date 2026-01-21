"""File extraction utilities for various file formats."""
import os
import logging
import fitz  # PyMuPDF
from docx import Document


MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB limit
MAX_TEXT_LENGTH = 50000  # Max characters to extract


def extract_text_from_file(file_path):
    """Extract text content from various file formats.
    
    Supports: PDF, DOCX, TXT, images (via OCR if available)
    
    Args:
        file_path: Path to the file
        
    Returns:
        Extracted text string, or empty string if extraction fails
    """
    if not os.path.exists(file_path):
        logging.warning("File not found: %s", file_path)
        return ""
    
    if not os.path.isfile(file_path):
        logging.warning("Not a file: %s", file_path)
        return ""
    
    # Check file size
    try:
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logging.debug("Skipping empty file: %s", file_path)
            return ""
        if file_size > MAX_FILE_SIZE:
            logging.warning("File too large (>100MB): %s", file_path)
            return ""
    except Exception as e:
        logging.warning("Error checking file size for %s: %s", file_path, e)
        return ""
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        # PDF files
        if file_ext == '.pdf':
            return extract_text_from_pdf(file_path)
        
        # Word documents
        elif file_ext == '.docx':
            return extract_text_from_docx(file_path)
        
        # Plain text
        elif file_ext == '.txt':
            return extract_text_from_txt(file_path)
        
        # Image files - attempt OCR
        elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return extract_text_from_image(file_path)
        
        # Default: try reading as text
        else:
            return extract_text_from_txt(file_path)
                
    except Exception as e:
        logging.error("Error extracting text from %s: %s", file_path, e)
        return ""


def extract_text_from_txt(file_path):
    """Extract text from plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(MAX_TEXT_LENGTH)
            return content.strip()
    except Exception as e:
        logging.warning("Error reading text file %s: %s", file_path, e)
        return ""


def extract_text_from_pdf(file_path):
    """Extract text from PDF file using PyMuPDF."""
    try:
        text = []
        doc = fitz.open(file_path)
        
        # Limit pages to first 50
        max_pages = min(len(doc), 50)
        for page_num in range(max_pages):
            try:
                page = doc[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    text.append(page_text)
            except Exception as e:
                logging.debug("Error extracting page %d from %s: %s", page_num, file_path, e)
                continue
        
        doc.close()
        result = "\n".join(text)
        return result[:MAX_TEXT_LENGTH]
    except Exception as e:
        logging.warning("Error extracting text from PDF %s: %s", file_path, e)
        return ""


def extract_text_from_docx(file_path):
    """Extract text from DOCX file using python-docx."""
    try:
        doc = Document(file_path)
        text = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                text.append(para.text)
        
        result = "\n".join(text)
        return result[:MAX_TEXT_LENGTH]
    except Exception as e:
        logging.warning("Error extracting text from DOCX %s: %s", file_path, e)
        return ""


def extract_text_from_image(file_path):
    """Extract text from image using Tesseract OCR if available."""
    try:
        import pytesseract
        from PIL import Image
        
        try:
            # Check if Tesseract is available
            pytesseract.get_tesseract_version()
            
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
            return text.strip()
        except pytesseract.TesseractNotFoundError:
            logging.debug("Tesseract not available for OCR")
            return ""
        except Exception as e:
            logging.debug("OCR failed for %s: %s", file_path, e)
            return ""
    except ImportError:
        logging.debug("PIL or pytesseract not available for OCR")
        return ""


def get_file_metadata(file_path):
    """Get basic metadata about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file metadata
    """
    try:
        stat = os.stat(file_path)
        return {
            'size': stat.st_size,
            'extension': os.path.splitext(file_path)[1],
            'name': os.path.basename(file_path),
            'modified': stat.st_mtime
        }
    except Exception as e:
        logging.warning("Error getting file metadata for %s: %s", file_path, e)
        return {}
