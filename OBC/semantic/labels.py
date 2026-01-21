import re
import logging
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from utils.text import normalize, clean_text

# Default classification labels
LABELS = [
    "Documents",
    "Spreadsheets",
    "Presentations",
    "Images",
    "Code",
    "Audio",
    "Video",
    "Archives",
    "Research",
    "Reports",
    "Data",
    "Documentation",
    "Academic",
    "Others"
]


class DynamicLabelGenerator:
    def __init__(self, n_topics=10, n_words=5, max_labels=15):
        self.n_topics = n_topics
        self.n_words = n_words
        self.max_labels = max_labels
        self.documents = []
        self.labels = set(LABELS)

        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.lda = LatentDirichletAllocation(
            n_components=min(n_topics, 5), 
            random_state=42,
            max_iter=10
        )

    def add_document(self, text):
        """Add a document to the label generator."""
        if isinstance(text, str) and text.strip():
            self.documents.append(clean_text(text.lower()))

    def extract_keywords(self, text):
        """Extract keywords from text using POS tagging."""
        try:
            tokens = word_tokenize(text.lower())
            stop = set(stopwords.words('english'))
            tokens = [t for t in tokens if t not in stop and len(t) > 2]

            pos = nltk.pos_tag(tokens)
            return [w for w, p in pos if p.startswith(('NN', 'JJ'))]
        except Exception as e:
            logging.warning("Error extracting keywords: %s", e)
            return []

    def generate_labels(self):
        """Generate dynamic labels from documents."""
        if not self.documents:
            return self.labels

        try:
            if len(self.documents) < 2:
                return self.labels
                
            matrix = self.vectorizer.fit_transform(self.documents)
            self.lda.fit(matrix)
            features = self.vectorizer.get_feature_names_out()

            topics = set()
            for topic_idx, topic in enumerate(self.lda.components_):
                idx = topic.argsort()[:-self.n_words-1:-1]
                topic_words = " ".join(str(features[i]) for i in idx)
                if topic_words.strip():
                    topics.add(topic_words.title())

            # Limit total labels
            all_labels = self.labels | topics
            if len(all_labels) > self.max_labels:
                return list(self.labels)[:self.max_labels]
            
            return all_labels

        except Exception as e:
            logging.error("Label generation failed: %s", e)
            return self.labels

