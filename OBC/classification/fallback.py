"""Fallback classification methods when ML models are unavailable."""
import os
import re
import logging
from collections import defaultdict


# File type to label mapping
FILE_TYPE_MAPPING = {
    # Documents
    '.pdf': 'Documents',
    '.doc': 'Documents',
    '.docx': 'Documents',
    '.txt': 'Documents',
    '.md': 'Documentation',
    '.rst': 'Documentation',
    
    # Spreadsheets
    '.xls': 'Spreadsheets',
    '.xlsx': 'Spreadsheets',
    '.csv': 'Spreadsheets',
    '.ods': 'Spreadsheets',
    
    # Presentations
    '.ppt': 'Presentations',
    '.pptx': 'Presentations',
    '.odp': 'Presentations',
    
    # Code
    '.py': 'Code',
    '.js': 'Code',
    '.java': 'Code',
    '.cpp': 'Code',
    '.c': 'Code',
    '.cs': 'Code',
    '.go': 'Code',
    '.rs': 'Code',
    '.ts': 'Code',
    '.jsx': 'Code',
    '.tsx': 'Code',
    '.html': 'Code',
    '.css': 'Code',
    '.json': 'Code',
    '.xml': 'Code',
    '.yaml': 'Code',
    '.yml': 'Code',
    '.sh': 'Code',
    '.bat': 'Code',
    
    # Images
    '.jpg': 'Images',
    '.jpeg': 'Images',
    '.png': 'Images',
    '.gif': 'Images',
    '.bmp': 'Images',
    '.svg': 'Images',
    '.tiff': 'Images',
    '.webp': 'Images',
    
    # Audio
    '.mp3': 'Audio',
    '.wav': 'Audio',
    '.flac': 'Audio',
    '.aac': 'Audio',
    '.m4a': 'Audio',
    
    # Video
    '.mp4': 'Video',
    '.avi': 'Video',
    '.mov': 'Video',
    '.mkv': 'Video',
    '.flv': 'Video',
    '.wmv': 'Video',
    
    # Archives
    '.zip': 'Archives',
    '.rar': 'Archives',
    '.7z': 'Archives',
    '.tar': 'Archives',
    '.gz': 'Archives',
    '.iso': 'Archives',
}


# Content keywords for classification
CONTENT_KEYWORDS = {
    'Code': [
        'function', 'def ', 'class ', 'return', 'import', 'export', 
        'const ', 'let ', 'var ', 'async', 'await', 'promise', 'if ',
        'for ', 'while ', 'switch', 'try', 'catch', 'throw', 'new ',
        'this.', 'super', 'constructor', 'interface', 'type ', 'enum',
        'main()', 'public class', 'private ', 'static ', 'void ', 'null'
    ],
    'Documentation': [
        'readme', 'guide', 'manual', 'tutorial', 'howto', 'documentation',
        'description', 'overview', 'introduction', 'getting started',
        'installation', 'setup', 'configuration', 'usage', 'example',
        'api documentation', 'user guide', 'technical guide'
    ],
    'Academic': [
        'abstract', 'introduction', 'methodology', 'conclusion', 'references',
        'research', 'study', 'analysis', 'hypothesis', 'experiment',
        'results', 'discussion', 'literature', 'citation', 'peer review',
        'journal', 'conference', 'thesis'
    ],
    'Reports': [
        'report', 'summary', 'findings', 'results', 'analysis', 'conclusion',
        'executive summary', 'overview', 'statistics', 'metrics', 'performance',
        'evaluation', 'assessment', 'quarterly', 'annual', 'monthly'
    ],
    'Data': [
        'dataset', 'csv', 'database', 'record', 'field', 'schema',
        'table', 'row', 'column', 'query', 'sql', 'json', 'xml',
        'values', 'entries', 'records', 'attribute'
    ],
}


def classify_by_file_type(file_path: str) -> tuple:
    """Classify based on file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        (label, confidence) tuple
    """
    try:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext in FILE_TYPE_MAPPING:
            label = FILE_TYPE_MAPPING[ext]
            return label, 0.95
        
        return "Others", 0.3
    except Exception as e:
        logging.warning("Error classifying by file type: %s", e)
        return "Others", 0.3


def classify_by_content(text: str) -> tuple:
    """Classify based on text content patterns.
    
    Args:
        text: Document text
        
    Returns:
        (label, confidence) tuple
    """
    if not text or len(text.strip()) < 10:
        return "Others", 0.3
    
    text_lower = text.lower()
    
    # Score each category
    scores = defaultdict(float)
    
    for category, keywords in CONTENT_KEYWORDS.items():
        for keyword in keywords:
            count = text_lower.count(keyword.lower())
            if count > 0:
                scores[category] += count
    
    if not scores:
        return "Others", 0.3
    
    # Find best match
    best_category = max(scores.items(), key=lambda x: x[1])
    category, score = best_category
    
    # Normalize score to confidence (0-1)
    # More keywords = higher confidence
    confidence = min(0.9, 0.3 + (score * 0.05))
    
    return category, confidence


def pattern_match_classification(text: str, labels: list, threshold: float = 0.2) -> tuple:
    """Classify using pattern matching against provided labels.
    
    Args:
        text: Document text
        labels: List of possible labels
        threshold: Minimum confidence threshold
        
    Returns:
        (label, confidence) tuple
    """
    if not text or len(text.strip()) < 10:
        return "Others", 0.3
    
    text_lower = text.lower()
    scores = defaultdict(float)
    
    # Simple pattern matching: exact label name matches
    for label in labels:
        label_lower = label.lower()
        if label_lower in text_lower:
            # Count occurrences
            count = text_lower.count(label_lower)
            scores[label] = count * 0.5
    
    if not scores or max(scores.values()) < threshold:
        return "Others", 0.3
    
    best_label = max(scores.items(), key=lambda x: x[1])
    label, score = best_label
    
    confidence = min(0.9, threshold + (score * 0.1))
    return label, confidence
