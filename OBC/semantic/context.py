"""Context and relationship-based document classification."""
import os
import logging
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class ContextualClassifier:
    """Classifies documents based on content context and relationships."""
    
    def __init__(self):
        self.document_cache = {}
        self.similarity_matrix = None
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.document_vectors = None
        
    def add_document(self, file_path, text):
        """Add a document to the context cache.
        
        Args:
            file_path: Path to the file
            text: Extracted text content
        """
        self.document_cache[file_path] = {
            'text': text,
            'filename': os.path.basename(file_path),
            'extension': os.path.splitext(file_path)[1].lower(),
            'dirname': os.path.dirname(file_path)
        }
    
    def find_similar_documents(self, file_path, threshold=0.3):
        """Find documents similar to the given file.
        
        Args:
            file_path: Path to query file
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of (file_path, similarity_score) tuples
        """
        if file_path not in self.document_cache:
            return []
        
        if not self.document_vectors:
            self._build_similarity_matrix()
        
        file_index = list(self.document_cache.keys()).index(file_path)
        similarities = self.similarity_matrix[file_index]
        
        similar = []
        for idx, score in enumerate(similarities):
            if score > threshold and idx != file_index:
                other_path = list(self.document_cache.keys())[idx]
                similar.append((other_path, float(score)))
        
        return sorted(similar, key=lambda x: x[1], reverse=True)
    
    def _build_similarity_matrix(self):
        """Build TF-IDF similarity matrix for all documents."""
        if not self.document_cache:
            return
        
        try:
            texts = [doc['text'] for doc in self.document_cache.values()]
            self.document_vectors = self.vectorizer.fit_transform(texts)
            self.similarity_matrix = cosine_similarity(self.document_vectors)
        except Exception as e:
            logging.warning("Could not build similarity matrix: %s", e)
            self.similarity_matrix = None
    
    def detect_theme(self, text):
        """Detect the primary theme/category from text content.
        
        Args:
            text: Document text
            
        Returns:
            (theme: str, confidence: float)
        """
        text_lower = text.lower()
        
        themes = {
            'Resume/CV': {
                'keywords': ['summary', 'experience', 'education', 'skills', 'employment', 'qualification'],
                'weight': 5
            },
            'Project': {
                'keywords': ['project', 'implementation', 'development', 'deliverable', 'milestone', 'scope'],
                'weight': 4
            },
            'Meeting Notes': {
                'keywords': ['meeting', 'agenda', 'attendees', 'action items', 'discussion', 'discussed'],
                'weight': 4
            },
            'Technical Docs': {
                'keywords': ['api', 'function', 'parameter', 'return', 'method', 'class', 'module', 'architecture'],
                'weight': 5
            },
            'Design': {
                'keywords': ['design', 'layout', 'ui', 'ux', 'wireframe', 'mockup', 'visual', 'aesthetic'],
                'weight': 4
            },
            'Data/Analysis': {
                'keywords': ['data', 'analysis', 'metrics', 'statistics', 'chart', 'trend', 'result', 'dataset'],
                'weight': 4
            },
            'Planning': {
                'keywords': ['plan', 'strategy', 'timeline', 'roadmap', 'goal', 'objective', 'target'],
                'weight': 4
            },
        }
        
        scores = {}
        for theme, config in themes.items():
            score = 0
            for keyword in config['keywords']:
                if keyword in text_lower:
                    score += config['weight'] * text_lower.count(keyword)
            scores[theme] = score
        
        if not scores or max(scores.values()) == 0:
            return 'General', 0.3
        
        best_theme = max(scores.items(), key=lambda x: x[1])
        theme, score = best_theme
        
        # Normalize confidence
        confidence = min(0.95, 0.4 + (score * 0.01))
        return theme, confidence
    
    def classify_by_context(self, file_path, text):
        """Classify file based on content context and relationships.
        
        Args:
            file_path: Path to the file
            text: Extracted text content
            
        Returns:
            (category: str, confidence: float, context_info: dict)
        """
        theme, theme_conf = self.detect_theme(text)
        
        # Find similar documents
        similar_docs = self.find_similar_documents(file_path, threshold=0.25)
        
        context_info = {
            'theme': theme,
            'theme_confidence': theme_conf,
            'related_files': similar_docs[:5],  # Top 5 similar files
            'total_cache_size': len(self.document_cache)
        }
        
        # Map theme to category
        theme_to_category = {
            'Resume/CV': 'Documents',
            'Project': 'Documentation',
            'Meeting Notes': 'Documents',
            'Technical Docs': 'Documentation',
            'Design': 'Images',
            'Data/Analysis': 'Data',
            'Planning': 'Documentation',
            'General': 'Others'
        }
        
        category = theme_to_category.get(theme, 'Others')
        
        return category, theme_conf, context_info
    
    def get_document_group(self, file_path):
        """Get all documents that should be grouped together.
        
        Args:
            file_path: Path to the reference file
            
        Returns:
            List of related file paths
        """
        similar = self.find_similar_documents(file_path, threshold=0.4)
        related = [file_path] + [path for path, _ in similar]
        return related
    
    def clear_cache(self):
        """Clear the document cache."""
        self.document_cache.clear()
        self.similarity_matrix = None
        self.document_vectors = None


# Global classifier instance
_context_classifier = ContextualClassifier()


def classify_with_context(file_path, text):
    """Classify document using contextual analysis.
    
    Args:
        file_path: Path to the file
        text: Extracted text content
        
    Returns:
        (category: str, confidence: float)
    """
    _context_classifier.add_document(file_path, text)
    category, confidence, context = _context_classifier.classify_by_context(file_path, text)
    
    logging.debug(
        "File %s classified as %s (theme: %s, confidence: %.2f)",
        os.path.basename(file_path),
        category,
        context['theme'],
        confidence
    )
    
    return category, confidence


def find_related_documents(file_path, threshold=0.4):
    """Find documents related to the given file.
    
    Args:
        file_path: Path to the reference file
        threshold: Similarity threshold
        
    Returns:
        List of related file paths with similarity scores
    """
    return _context_classifier.find_similar_documents(file_path, threshold=threshold)
