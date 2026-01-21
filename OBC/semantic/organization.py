"""Dynamic folder generation and file clustering based on semantic relationships."""
import os
import logging
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist, squareform
import numpy as np
from utils.text import extract_keywords


class HierarchicalOrganizer:
    """Creates hierarchical folder structures with subfolders based on context."""
    
    # Context-specific subfolder definitions
    CONTEXT_SUBFOLDERS = {
        'Resume/CV': ['Resume_Files', 'Cover_Letters', 'Application_Materials', 'Interview_Prep'],
        'Project': ['Code', 'Documentation', 'Resources', 'Tests', 'Artifacts'],
        'Meeting Notes': ['Agendas', 'Notes', 'Action_Items', 'Decisions'],
        'Technical Docs': ['API_Reference', 'Guides', 'Architecture', 'Examples'],
        'Data/Analysis': ['Raw_Data', 'Processed_Data', 'Analysis', 'Visualizations'],
        'Design': ['Mockups', 'Prototypes', 'Assets', 'Guidelines'],
        'Planning': ['Requirements', 'Timelines', 'Budgets', 'Resources'],
    }
    
    def __init__(self):
        self.documents = []
        self.project_groups = {}
        self.context_groups = {}
    
    def organize_hierarchically(self, documents_info):
        """Create hierarchical folder structure with subfolders.
        
        Args:
            documents_info: List of dicts with keys: 'path', 'text', 'theme', 'filename'
            
        Returns:
            Dict mapping 'ParentFolder/SubFolder' to list of file paths
        """
        self.documents = documents_info
        structure = {}
        processed = set()
        
        # Phase 1: Group related files (detect projects, topics)
        grouped = self._group_by_context(documents_info)
        
        # Phase 2: Create hierarchical structure with subfolders
        for context_group, files in grouped.items():
            for file_path in files:
                if file_path in processed:
                    continue
                
                # Find which subfolder this file belongs to
                file_doc = next((d for d in documents_info if d['path'] == file_path), None)
                if not file_doc:
                    continue
                
                theme = file_doc.get('theme', 'Uncategorized')
                subfolder_name = self._determine_subfolder(file_path, file_doc, theme)
                
                # Create hierarchical path
                parent_folder = context_group
                if subfolder_name:
                    hierarchical_path = f"{parent_folder}/{subfolder_name}"
                else:
                    hierarchical_path = parent_folder
                
                if hierarchical_path not in structure:
                    structure[hierarchical_path] = []
                
                structure[hierarchical_path].append(file_path)
                processed.add(file_path)
        
        return structure
    
    def _group_by_context(self, documents_info):
        """Group files by detected context (projects, job search, etc).
        
        Returns:
            Dict mapping context name to list of file paths
        """
        groups = defaultdict(list)
        
        for doc_info in documents_info:
            file_path = doc_info['path']
            text = doc_info.get('text', '')
            theme = doc_info.get('theme', '')
            filename = doc_info.get('filename', '')
            
            # Detect job search related files
            if self._is_job_search_file(text, filename, theme):
                groups['Job_Search'].append(file_path)
            
            # Detect project files by analyzing text for project keywords
            elif self._detect_project(text, filename):
                project_name = self._extract_project_name(text, filename)
                groups[project_name].append(file_path)
            
            # Group by theme if no specific project detected
            else:
                theme_folder = theme.replace(' ', '_') if theme else 'Miscellaneous'
                groups[theme_folder].append(file_path)
        
        return groups
    
    def _is_job_search_file(self, text, filename, theme):
        """Check if file is job search related."""
        job_keywords = [
            'resume', 'cv', 'cover letter', 'job', 'application', 'position',
            'hiring', 'interview', 'candidate', 'employment', 'recruiter',
            'linkedin', 'portfolio', 'references', 'experience', 'qualifications'
        ]
        
        text_lower = text.lower()
        filename_lower = filename.lower()
        theme_lower = theme.lower()
        
        # Check filename first (highest confidence)
        for keyword in job_keywords[:3]:  # resume, cv, cover letter
            if keyword in filename_lower:
                return True
        
        # Check text content
        keyword_matches = sum(1 for kw in job_keywords if kw in text_lower)
        if keyword_matches >= 3:
            return True
        
        # Check theme
        if 'resume' in theme_lower or 'cv' in theme_lower:
            return True
        
        return False
    
    def _detect_project(self, text, filename):
        """Detect if file is part of a specific project."""
        project_keywords = [
            'project', 'sprint', 'milestone', 'deliverable', 'task',
            'repository', 'github', 'gitlab', 'branch', 'commit',
            'build', 'deployment', 'release', 'version', 'module'
        ]
        
        text_lower = text.lower()
        keyword_count = sum(1 for kw in project_keywords if kw in text_lower)
        
        return keyword_count >= 2
    
    def _extract_project_name(self, text, filename):
        """Extract project name from content."""
        # Look for project name patterns in text
        project_keywords = ['project:', 'projectname:', 'project name:', 'project id:']
        text_lower = text.lower()
        
        for kw in project_keywords:
            if kw in text_lower:
                start = text_lower.index(kw) + len(kw)
                end = start + 50
                extracted = text[start:end].split('\n')[0].strip()
                if extracted and len(extracted) < 30:
                    return f"Project_{extracted.replace(' ', '_')}"
        
        # Extract from filename
        name_part = filename.split('.')[0]
        if len(name_part) > 3 and len(name_part) < 50:
            return f"Project_{name_part.replace(' ', '_')}"
        
        return "Project_Misc"
    
    def _determine_subfolder(self, file_path, file_doc, theme):
        """Determine appropriate subfolder based on file content and type."""
        filename = file_doc.get('filename', '')
        text = file_doc.get('text', '')
        
        # Get subfolders for this theme
        subfolders = self.CONTEXT_SUBFOLDERS.get(theme, [])
        if not subfolders:
            return None
        
        # Match file to subfolder based on content
        filename_lower = filename.lower()
        text_lower = text.lower()
        
        # Define keywords for each subfolder type
        subfolder_keywords = {
            'Resume_Files': ['resume', 'cv', 'curriculum', 'vitae'],
            'Cover_Letters': ['cover', 'letter', 'application'],
            'Application_Materials': ['application', 'form', 'questionnaire'],
            'Interview_Prep': ['interview', 'preparation', 'question', 'answer'],
            'Code': ['code', 'script', 'function', 'class', 'def', 'import', '.py', '.js', '.java'],
            'Documentation': ['readme', 'guide', 'manual', 'tutorial', 'documentation'],
            'Resources': ['resource', 'reference', 'library', 'tool'],
            'Tests': ['test', 'unit', 'integration', 'test_'],
            'Artifacts': ['artifact', 'build', 'release', 'output'],
            'Agendas': ['agenda', 'schedule', 'time'],
            'Notes': ['note', 'minutes', 'summary'],
            'Action_Items': ['action', 'todo', 'task', 'next'],
            'Decisions': ['decision', 'resolution', 'approved'],
            'API_Reference': ['api', 'endpoint', 'method', 'parameter'],
            'Guides': ['guide', 'tutorial', 'how', 'step'],
            'Architecture': ['architecture', 'design', 'diagram', 'structure'],
            'Examples': ['example', 'sample', 'demo'],
            'Raw_Data': ['raw', 'original', 'source', 'csv', 'xlsx'],
            'Processed_Data': ['processed', 'cleaned', 'transformed'],
            'Analysis': ['analysis', 'analytics', 'analyze', 'report'],
            'Visualizations': ['chart', 'graph', 'visualization', 'plot', 'image'],
            'Mockups': ['mockup', 'wireframe', 'sketch'],
            'Prototypes': ['prototype', 'proof', 'poc'],
            'Assets': ['asset', 'icon', 'image', 'svg', 'png'],
            'Guidelines': ['guideline', 'standard', 'style', 'brand'],
            'Requirements': ['requirement', 'spec', 'specification'],
            'Timelines': ['timeline', 'schedule', 'gantt', 'milestone'],
            'Budgets': ['budget', 'cost', 'expense', 'pricing'],
        }
        
        # Score each subfolder
        best_match = None
        best_score = 0
        
        for subfolder, keywords in subfolder_keywords.items():
            if subfolder not in subfolders:
                continue
            
            score = 0
            for keyword in keywords:
                if keyword in filename_lower:
                    score += 3
                if keyword in text_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = subfolder
        
        # Return best match or first subfolder as default
        return best_match if best_score > 0 else subfolders[0]


class DynamicFolderGenerator:
    """Generates folder names and clusters files based on semantic relationships."""
    
    def __init__(self, min_cluster_size=2, similarity_threshold=0.3):
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        self.documents = []
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.tfidf_matrix = None
    
    def add_document(self, file_path, text, theme=''):
        """Add a document for clustering.
        
        Args:
            file_path: Path to the file
            text: Extracted text content
            theme: Detected theme/category
        """
        self.documents.append({
            'path': file_path,
            'filename': os.path.basename(file_path),
            'text': text,
            'theme': theme
        })
    
    def cluster_documents(self):
        """Cluster documents based on semantic similarity.
        
        Returns:
            Dict mapping cluster_id to list of document paths
        """
        if len(self.documents) < 2:
            # Each single file gets its own cluster
            return {i: [doc['path']] for i, doc in enumerate(self.documents)}
        
        try:
            # Build TF-IDF matrix
            texts = [doc['text'] for doc in self.documents]
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate distance matrix
            distance_matrix = 1 - (self.tfidf_matrix @ self.tfidf_matrix.T).toarray()
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(
                eps=1 - self.similarity_threshold,
                min_samples=1,
                metric='precomputed'
            ).fit(distance_matrix)
            
            # Group by cluster
            clusters = defaultdict(list)
            for doc_idx, cluster_id in enumerate(clustering.labels_):
                clusters[cluster_id].append(self.documents[doc_idx]['path'])
            
            return dict(clusters)
        
        except Exception as e:
            logging.warning("Clustering failed: %s, using themes instead", e)
            return self._cluster_by_theme()
    
    def _cluster_by_theme(self):
        """Fallback clustering by theme."""
        clusters = defaultdict(list)
        for i, doc in enumerate(self.documents):
            theme_key = doc['theme'] or 'Uncategorized'
            clusters[theme_key].append(doc['path'])
        return dict(clusters)
    
    def generate_folder_name(self, cluster_files):
        """Generate a meaningful folder name for a cluster.
        
        Args:
            cluster_files: List of file paths in the cluster
            
        Returns:
            Folder name string
        """
        if not cluster_files:
            return "Unnamed_Folder"
        
        # Get documents for this cluster
        cluster_docs = [doc for doc in self.documents if doc['path'] in cluster_files]
        
        # Common themes in cluster
        themes = [doc['theme'] for doc in cluster_docs if doc['theme']]
        theme_counts = defaultdict(int)
        for theme in themes:
            theme_counts[theme] += 1
        
        # Get most common theme
        if theme_counts:
            primary_theme = max(theme_counts.items(), key=lambda x: x[1])[0]
            return self._theme_to_folder_name(primary_theme, cluster_docs)
        
        # Generate from filenames and content
        return self._generate_from_content(cluster_docs)
    
    def _theme_to_folder_name(self, theme, documents):
        """Convert theme to folder name.
        
        Args:
            theme: Theme string
            documents: List of documents
            
        Returns:
            Folder name
        """
        theme_mapping = {
            'Resume/CV': 'Resume_Profile',
            'Project': 'Project_Files',
            'Meeting Notes': 'Meeting_Records',
            'Technical Docs': 'Technical_Documentation',
            'Design': 'Design_Assets',
            'Data/Analysis': 'Data_Analysis',
            'Planning': 'Planning_Strategy',
        }
        
        folder_name = theme_mapping.get(theme, theme.replace(' ', '_'))
        
        # Add file count suffix if needed
        if len(documents) > 1:
            folder_name = f"{folder_name}_{len(documents)}files"
        
        return folder_name
    
    def _generate_from_content(self, documents):
        """Generate folder name from document content.
        
        Args:
            documents: List of document dicts
            
        Returns:
            Generated folder name
        """
        # Extract key terms from first document
        main_doc = documents[0]
        text_lower = main_doc['text'].lower()
        
        # Common key terms
        key_terms = [
            'project', 'analysis', 'report', 'summary', 'proposal',
            'document', 'review', 'planning', 'research', 'study'
        ]
        
        for term in key_terms:
            if term in text_lower:
                return f"{term.title()}_Group"
        
        # Use filename if no theme found
        filename = main_doc['filename'].split('.')[0]
        return f"{filename}_Group" if len(documents) > 1 else filename
    
    def get_organization_structure(self):
        """Get the complete organization structure.
        
        Returns:
            Dict mapping folder_name to list of file paths
        """
        clusters = self.cluster_documents()
        structure = {}
        
        for cluster_id, files in clusters.items():
            folder_name = self.generate_folder_name(files)
            # Ensure unique folder names
            unique_name = folder_name
            counter = 1
            while unique_name in structure:
                unique_name = f"{folder_name}_{counter}"
                counter += 1
            
            structure[unique_name] = files
        
        return structure
    
    def clear(self):
        """Clear all cached documents."""
        self.documents = []
        self.tfidf_matrix = None


# Global instance
_folder_generator = DynamicFolderGenerator()


def get_organization_structure(documents_info):
    """Generate hierarchical folder structure from documents.
    
    Args:
        documents_info: List of dicts with keys: 'path', 'text', 'theme', 'filename'
        
    Returns:
        Dict mapping 'ParentFolder' or 'ParentFolder/SubFolder' to list of file paths
    """
    # Use hierarchical organizer for better grouping
    organizer = HierarchicalOrganizer()
    return organizer.organize_hierarchically(documents_info)


def generate_folder_name_for_cluster(file_paths):
    """Generate a folder name for specific files.
    
    Args:
        file_paths: List of file paths
        
    Returns:
        Folder name string
    """
    return _folder_generator.generate_folder_name(file_paths)
