# Dynamic Content-Based Organization

## Overview

The Smart File Organizer now creates **intelligent folder structures** based on **semantic relationships and content context** rather than fixed categories.

## How It Works

### 3-Phase Processing:

**Phase 1: Analysis**
- Extracts text from all files
- Classifies each file by content theme (Resume, Meeting Notes, Technical Docs, etc.)
- Caches document information

**Phase 2: Clustering**
- Uses TF-IDF vectorization to analyze semantic similarity
- Groups related documents together using DBSCAN clustering
- Finds documents that belong together conceptually

**Phase 3: Organization**
- Generates intelligent folder names based on cluster themes
- Moves files into dynamically created folders
- Clean folder structure reflecting actual content relationships

## Example Organization

### Before (File-Type Based):
```
sorted_files/
├── Documents/
│   ├── resume.docx
│   ├── cover_letter.docx
│   ├── meeting_notes.docx
│   └── project_plan.docx
├── Code/
├── Images/
└── Others/
```

### After (Content-Relationship Based):
```
sorted_files/
├── Resume_Profile_2files/
│   ├── resume.docx
│   └── cover_letter.docx
├── Project_Files_2files/
│   ├── project_plan.docx
│   └── project_timeline.docx
└── Meeting_Records/
    └── meeting_notes.docx
```

## Features

✅ **Intelligent Clustering** - Finds documents that belong together
✅ **Context-Aware Naming** - Folder names reflect content purpose
✅ **Relationship Detection** - Understands document relationships
✅ **Dynamic Structure** - No predefined folder categories
✅ **Semantic Similarity** - Groups by meaning, not just metadata
✅ **Flexible Thresholds** - Adjustable clustering sensitivity

## Theme Detection

The system recognizes and organizes by:

- **Resume/CV** → Resume_Profile
- **Project** → Project_Files
- **Meeting Notes** → Meeting_Records
- **Technical Docs** → Technical_Documentation
- **Design** → Design_Assets
- **Data/Analysis** → Data_Analysis
- **Planning** → Planning_Strategy

## Configuration

### To adjust clustering sensitivity:

Edit [semantic/organization.py](semantic/organization.py):

```python
# In DynamicFolderGenerator.__init__():
min_cluster_size=2          # Minimum files per cluster
similarity_threshold=0.3    # 0.3 = more aggressive clustering
```

Lower threshold = larger clusters
Higher threshold = more granular clustering

## Example Use Case

**Your Documents:**
1. resume.docx - Contains: skills, experience, education
2. cover_letter.docx - References resume, discusses application
3. meeting_notes.docx - Project discussion, action items
4. architecture.pdf - Technical design, components
5. api_docs.docx - API reference, documentation

**Result:**
```
├── Resume_Profile_2files/      (resume + cover letter grouped)
├── Project_Meeting_2files/     (meeting notes + related discussion)
└── Technical_Documentation/    (architecture + api docs)
```

## How Similarity Works

The system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to:
1. Extract key terms from each document
2. Calculate semantic distance between all pairs
3. Group documents with similar vocabulary and themes
4. Generate names from the dominant theme in each group

## Benefits

✓ Related files stay together automatically
✓ No manual pre-configuration needed
✓ Learns from your document patterns
✓ Scales to any number of files
✓ Works with any file format
✓ Preserves document context through grouping
