import logging
import threading
import torch
from transformers import pipeline

from classification.fallback import classify_by_content, classify_by_file_type
from semantic.labels import LABELS
from semantic.context import classify_with_context
from utils.text import clean_text

CONFIDENCE_THRESHOLD = 0.5
MIN_TEXT_LENGTH = 30

_document_classifier = None
_model_lock = threading.Lock()
_use_context = True  # Enable context-based classification

# -------------------------------------------------
# Model Initialization
# -------------------------------------------------

def initialize_model():
    global _document_classifier

    if _document_classifier is not None:
        return

    with _model_lock:
        if _document_classifier is not None:
            return

        device = 0 if torch.cuda.is_available() else -1
        logging.info("Initializing zero-shot classifier (device=%s)", device)

        try:
            _document_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device
            )
        except Exception as e:
            logging.error("Failed to load transformer model: %s", e)
            _document_classifier = None

# -------------------------------------------------
# Classification
# -------------------------------------------------

def classify_document(text: str, file_path: str = None):
    """Classify document based on content context and relationships.
    
    Args:
        text: Document text content
        file_path: Optional file path for context analysis

    Returns:
        (label: str, confidence: float)
    """
    # File-type detection takes priority for known types - high confidence
    if file_path:
        label, conf = classify_by_file_type(file_path)
        if conf >= 0.90:  # High confidence from file type
            return label, conf
    
    # If text is too short or empty, use file-type fallback
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        if file_path:
            label, conf = classify_by_file_type(file_path)
            if label != "Others":
                return label, conf
        return "Others", 0.3

    text = clean_text(text)
    
    # Try context-based classification if enabled and we have a file path
    if _use_context and file_path:
        try:
            label, conf = classify_with_context(file_path, text)
            if conf >= CONFIDENCE_THRESHOLD:
                return label, conf
        except Exception as e:
            logging.debug("Context classification failed: %s", e)
    
    initialize_model()

    if _document_classifier is None:
        # Use content-based fallback
        return classify_by_content(text)

    try:
        result = _document_classifier(text, list(LABELS), multi_label=False)
        score = result["scores"][0]
        label = result["labels"][0]

        if score >= CONFIDENCE_THRESHOLD:
            return label, float(score)
        else:
            return classify_by_content(text)

    except Exception as e:
        logging.warning("Transformer classification failed: %s", e)
        return classify_by_content(text)
