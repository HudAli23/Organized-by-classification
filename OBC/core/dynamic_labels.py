import os
import json
import logging
from collections import defaultdict
import re
from typing import Set, Dict, List, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class DynamicLabelGenerator:
    """A class to dynamically generate and manage document labels based on content analysis."""
    
    def __init__(self):
        self.document_count = 0
        self.word_frequencies = defaultdict(int)
        self.label_scores = defaultdict(float)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.processed_tokens_cache = {}  # Cache for preprocessed tokens
        self.label_cache = {}  # Cache for generated labels
        
        # Load or create the label history file
        self.history_file = "label_history.json"
        self.load_label_history()
        
        # Initialize base categories with optimized structure
        self.base_categories = {
            "Documents": {"weight": 1.0, "keywords": set(["document", "text", "letter", "report"])},
            "Images": {"weight": 1.0, "keywords": set(["image", "photo", "picture", "graphic"])},
            "Audio": {"weight": 1.0, "keywords": set(["audio", "sound", "music", "recording"])},
            "Video": {"weight": 1.0, "keywords": set(["video", "movie", "film", "animation"])},
            "Code": {"weight": 1.0, "keywords": set(["code", "programming", "script", "software"])},
            "Data": {"weight": 1.0, "keywords": set(["data", "dataset", "database", "spreadsheet"])},
            "Archives": {"weight": 1.0, "keywords": set(["archive", "backup", "compressed", "zip"])}
        }
        
        # Compile regex patterns for better performance
        self.domain_patterns = {
            "Machine Learning": [re.compile(pattern, re.IGNORECASE) for pattern in [
                r"\.model$", r"\.h5$", r"\.pkl$", r"train.*data",
                r"test.*data", r"validation", r"epoch", r"neural",
                r"tensorflow", r"pytorch", r"keras", r"scikit"
            ]],
            "Web Development": [re.compile(pattern, re.IGNORECASE) for pattern in [
                r"\.html$", r"\.css$", r"\.js$", r"\.php$",
                r"webpack", r"react", r"angular", r"vue",
                r"node_modules", r"package\.json"
            ]],
            "Database": [re.compile(pattern, re.IGNORECASE) for pattern in [
                r"\.sql$", r"\.db$", r"\.sqlite$",
                r"query", r"schema", r"table", r"database"
            ]],
            "Documentation": [re.compile(pattern, re.IGNORECASE) for pattern in [
                r"\.md$", r"\.rst$", r"\.txt$", r"readme",
                r"docs?", r"manual", r"guide", r"reference"
            ]]
        }
        
    def load_label_history(self):
        """Load label history from file or create if not exists."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.word_frequencies = defaultdict(int, data.get('word_frequencies', {}))
                    self.label_scores = defaultdict(float, data.get('label_scores', {}))
                    self.document_count = data.get('document_count', 0)
            else:
                self.save_label_history()
        except Exception as e:
            logging.error(f"Error loading label history: {e}")
            self.word_frequencies = defaultdict(int)
            self.label_scores = defaultdict(float)
            self.document_count = 0
            
    def save_label_history(self):
        """Save current label history to file."""
        try:
            data = {
                'word_frequencies': dict(self.word_frequencies),
                'label_scores': dict(self.label_scores),
                'document_count': self.document_count
            }
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving label history: {e}")
            
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text by tokenizing, removing stopwords, and lemmatizing."""
        # Check cache first
        if text in self.processed_tokens_cache:
            return self.processed_tokens_cache[text]
            
        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens in a single pass
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalpha() and token not in self.stop_words
        ]
        
        # Cache the result
        self.processed_tokens_cache[text] = tokens
        
        # Limit cache size
        if len(self.processed_tokens_cache) > 1000:
            # Remove oldest entries
            keys_to_remove = list(self.processed_tokens_cache.keys())[:-500]
            for key in keys_to_remove:
                del self.processed_tokens_cache[key]
        
        return tokens
        
    def add_document(self, text: str):
        """Process a new document and update word frequencies."""
        if not text:
            return
            
        # Preprocess text
        tokens = self.preprocess_text(text)
        
        # Update word frequencies in batch
        for token in set(tokens):  # Use set to count each token once per document
            self.word_frequencies[token] += 1
            
        # Update document count
        self.document_count += 1
        
        # Clear label cache since frequencies changed
        self.label_cache.clear()
        
        # Update label scores (but not too frequently)
        if self.document_count % 10 == 0:  # Update scores every 10 documents
            self._update_label_scores()
            self.save_label_history()
        
    def _update_label_scores(self):
        """Update label scores based on word frequencies and patterns."""
        # Reset scores
        self.label_scores.clear()
        
        # Calculate scores for base categories efficiently
        doc_count = max(1, self.document_count)  # Avoid division by zero
        
        for category, info in self.base_categories.items():
            # Calculate score in a single pass
            score = sum(
                (self.word_frequencies[keyword] / doc_count) * info["weight"]
                for keyword in info["keywords"]
                if keyword in self.word_frequencies
            )
            if score > 0:
                self.label_scores[category] = score
            
        # Calculate domain pattern scores efficiently
        for domain, patterns in self.domain_patterns.items():
            matching_words = sum(
                1 for word in self.word_frequencies
                if any(pattern.search(word) for pattern in patterns)
            )
            if matching_words > 0:
                self.label_scores[domain] = matching_words / len(patterns)
            
    def generate_labels(self, min_score: float = 0.1, max_labels: int = 15) -> Set[str]:
        """Generate a set of labels based on current scores."""
        # Check cache first
        cache_key = (min_score, max_labels)
        if cache_key in self.label_cache:
            return self.label_cache[cache_key].copy()
            
        if self.document_count == 0:
            return set(self.base_categories.keys())
            
        # Get labels with scores above threshold efficiently
        valid_labels = {
            label for label, score in self.label_scores.items()
            if score >= min_score
        }
        
        # Always include base categories
        valid_labels.update(self.base_categories.keys())
        
        # Limit number of labels efficiently
        if len(valid_labels) > max_labels:
            valid_labels = set(
                sorted(
                    valid_labels,
                    key=lambda x: self.label_scores.get(x, 0),
                    reverse=True
                )[:max_labels]
            )
            
        # Cache the result
        self.label_cache[cache_key] = valid_labels.copy()
        
        return valid_labels
        
    def get_label_scores(self) -> Dict[str, float]:
        """Return current label scores."""
        return dict(self.label_scores)
        
    def suggest_label(self, text: str) -> Optional[str]:
        """Suggest a single label for the given text."""
        if not text:
            return None
            
        # Process text
        tokens = set(self.preprocess_text(text))  # Use set for O(1) lookup
        
        # Calculate scores efficiently
        scores = defaultdict(float)
        
        # Check base categories
        for category, info in self.base_categories.items():
            # Calculate intersection between tokens and keywords
            matching_keywords = tokens.intersection(info["keywords"])
            if matching_keywords:
                scores[category] = len(matching_keywords) * info["weight"]
            
        # Check domain patterns efficiently
        text_lower = text.lower()
        for domain, patterns in self.domain_patterns.items():
            # Use any() for short-circuit evaluation
            if any(pattern.search(text_lower) for pattern in patterns):
                scores[domain] = 1.0
            
        # Return highest scoring category
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return None 