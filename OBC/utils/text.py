"""Text processing utilities for document analysis."""
import re
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def clean_text(text):
    """Clean and normalize text for analysis.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned and normalized text string
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def normalize(text):
    """Alias for clean_text for backward compatibility."""
    return clean_text(text)


def extract_keywords(text, num_keywords=10):
    """Extract key keywords from text using TF-IDF style scoring.
    
    Args:
        text: Text to extract keywords from
        num_keywords: Number of keywords to return
        
    Returns:
        List of keyword strings
    """
    try:
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        
        # Filter out stopwords and short tokens
        keywords = [
            token for token in tokens 
            if token.isalnum() and token not in stop_words and len(token) > 2
        ]
        
        # Return most frequent keywords
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [kw for kw, _ in keyword_counts.most_common(num_keywords)]
    except Exception as e:
        logging.warning("Error extracting keywords: %s", e)
        return []


def calculate_text_statistics(text):
    """Calculate basic text statistics.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with word count, char count, etc.
    """
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    return {
        'word_count': len(words),
        'char_count': len(text),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': len(text) / len(words) if words else 0
    }
