import os
import logging
import platform
import tkinter as tk
from logging.handlers import RotatingFileHandler

import nltk
import pytesseract

from file_organizer_gui import FileSorterApp

# -------------------- Logging --------------------

def setup_logging():
    """Configure logging with both file and console handlers."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.handlers:
        return

    os.makedirs('logs', exist_ok=True)

    file_handler = RotatingFileHandler(
        os.path.join('logs', 'file_organizer.log'),
        maxBytes=1_048_576,  # 1MB
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s: %(message)s'
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# -------------------- NLTK --------------------

def setup_nltk():
    """Ensure all required NLTK resources are available."""
    required_resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
    ]

    logging.info('Checking NLTK resources')

    for resource_path, resource_name in required_resources:
        try:
            nltk.data.find(resource_path)
            logging.info('NLTK resource available: %s', resource_name)
        except LookupError:
            try:
                logging.info('Downloading NLTK resource: %s', resource_name)
                nltk.download(resource_name, quiet=True)
            except Exception as e:
                logging.warning(
                    'Failed to download NLTK resource %s: %s',
                    resource_name,
                    e
                )

# -------------------- Tesseract --------------------

def setup_tesseract():
    """Locate and configure Tesseract OCR if available."""
    system = platform.system()
    possible_paths = []

    if system == 'Windows':
        possible_paths = [
            r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe',
            r'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe',
        ]
    elif system == 'Linux':
        possible_paths = ['/usr/bin/tesseract', '/usr/local/bin/tesseract']
    elif system == 'Darwin':
        possible_paths = ['/usr/local/bin/tesseract', '/opt/homebrew/bin/tesseract']

    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            logging.info('Tesseract found at: %s', path)
            return True

    logging.warning('Tesseract not found. OCR features will be disabled.')
    return False

# -------------------- Main --------------------

def main():
    """Main entry point of the application."""
    setup_logging()
    logging.info('Starting Smart File Organizer')

    # Optional dependencies
    setup_nltk()
    setup_tesseract()

    # Start GUI
    root = tk.Tk()
    app = FileSorterApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
