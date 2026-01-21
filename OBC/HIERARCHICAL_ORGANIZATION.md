# Hierarchical File Organization System

## Overview

Your file organizer now creates **intelligent hierarchical folder structures** where:
- **Each major context gets its own top-level folder** (Project_Alpha, Job_Search, etc.)
- **Each top-level folder contains subfolders** for specific file types and purposes
- **Files are automatically sorted into the most relevant subfolders** based on content analysis

This creates a clean, nested structure that reflects how files are actually related and used.

---

## Example Folder Structures

### Job Search Context
```
Destination/
└── Job_Search/
    ├── Resume_Files/
    │   ├── resume_v1.pdf
    │   ├── resume_final.docx
    │   └── resume_updated.pdf
    ├── Cover_Letters/
    │   ├── cover_letter_acme.docx
    │   └── cover_letter_techcorp.docx
    ├── Application_Materials/
    │   ├── application_form.pdf
    │   └── supplementary_answers.docx
    └── Interview_Prep/
        ├── interview_questions.txt
        ├── company_research.md
        └── behavioral_prep.docx
```

### Project Context
```
Destination/
└── Project_WebApp/
    ├── Code/
    │   ├── main.py
    │   ├── utils.py
    │   └── api.py
    ├── Documentation/
    │   ├── README.md
    │   ├── ARCHITECTURE.md
    │   └── API_REFERENCE.md
    ├── Resources/
    │   ├── logo.png
    │   ├── database_schema.sql
    │   └── wireframes.pdf
    ├── Tests/
    │   ├── test_main.py
    │   └── test_utils.py
    └── Artifacts/
        ├── build.zip
        └── release_v1.0.tar.gz
```

### Meeting Notes Context
```
Destination/
└── Meeting_Records/
    ├── Agendas/
    │   ├── Q1_Planning_Meeting_Agenda.docx
    │   └── Weekly_Standup_Agenda.txt
    ├── Notes/
    │   ├── 2025_01_15_Team_Meeting_Notes.md
    │   └── 2025_01_10_Strategy_Notes.docx
    ├── Action_Items/
    │   ├── Q1_Action_Items.xlsx
    │   └── Assigned_Tasks.md
    └── Decisions/
        ├── Technology_Stack_Decision.md
        └── Budget_Approval_Decision.md
```

### Data Analysis Context
```
Destination/
└── Data_Analysis/
    ├── Raw_Data/
    │   ├── sales_2025.csv
    │   ├── customer_database.xlsx
    │   └── logs.json
    ├── Processed_Data/
    │   ├── cleaned_sales.csv
    │   └── aggregated_metrics.xlsx
    ├── Analysis/
    │   ├── Q1_Performance_Report.pdf
    │   ├── trend_analysis.md
    │   └── quarterly_summary.xlsx
    └── Visualizations/
        ├── sales_chart.png
        ├── trend_graph.pdf
        └── dashboard_screenshot.png
```

---

## How It Works

### 1. **Content Analysis Phase**
The system reads each file and analyzes:
- **Filename** (e.g., "resume.docx" → Job Search context)
- **Content keywords** (e.g., "cover letter", "application" → Job Search)
- **Detected theme** (e.g., "Resume/CV", "Project", "Meeting Notes")

### 2. **Grouping Phase**
Files are grouped by detected context:
- **Job Search files** → All job-related files go to `Job_Search/`
- **Project files** → All files for a project go to `Project_[Name]/`
- **Meeting files** → All meeting-related files go to `Meeting_Records/`
- **Data files** → All data/analysis files go to `Data_Analysis/`
- etc.

### 3. **Subfolder Distribution Phase**
Within each context folder, files are sorted into subfolders based on their specific type:
- Resume files → `Resume_Files/`
- Cover letters → `Cover_Letters/`
- Code files → `Code/`
- Documentation → `Documentation/`
- etc.

---

## Context Types & Subfolders

### Job/Career Context
| Subfolder | Contains |
|-----------|----------|
| Resume_Files | Resumes, CVs, curriculum vitae |
| Cover_Letters | Cover letters, application letters |
| Application_Materials | Application forms, questionnaires |
| Interview_Prep | Interview questions, company research, preparation notes |

### Project Context
| Subfolder | Contains |
|-----------|----------|
| Code | Source code, scripts, implementations |
| Documentation | READMEs, guides, architecture docs |
| Resources | Assets, databases, configurations |
| Tests | Test files, test suites, test cases |
| Artifacts | Build outputs, releases, packages |

### Meeting Context
| Subfolder | Contains |
|-----------|----------|
| Agendas | Meeting agendas, outlines |
| Notes | Meeting minutes, notes, summaries |
| Action_Items | Task lists, assigned items, to-dos |
| Decisions | Decisions, resolutions, approvals |

### Technical Documentation Context
| Subfolder | Contains |
|-----------|----------|
| API_Reference | API docs, endpoints, specifications |
| Guides | Tutorials, how-to guides, step-by-step |
| Architecture | Architecture docs, system diagrams |
| Examples | Code examples, sample files, demos |

### Data Analysis Context
| Subfolder | Contains |
|-----------|----------|
| Raw_Data | Original data, CSV, Excel, JSON |
| Processed_Data | Cleaned data, transformed data |
| Analysis | Reports, analysis documents |
| Visualizations | Charts, graphs, images, dashboards |

### Design Context
| Subfolder | Contains |
|-----------|----------|
| Mockups | Wireframes, mockups, sketches |
| Prototypes | Prototypes, proof-of-concept |
| Assets | Icons, images, vectors, SVG |
| Guidelines | Style guides, brand guidelines |

### Planning Context
| Subfolder | Contains |
|-----------|----------|
| Requirements | Requirements, specifications |
| Timelines | Schedules, Gantt charts, milestones |
| Budgets | Budget files, cost estimates |

---

## Smart Detection Examples

### Example 1: Resume Detection
**File:** `my_resume.pdf`
**Content keywords:** "professional experience", "education", "skills", "employment"
**Result:** → `Job_Search/Resume_Files/my_resume.pdf`

### Example 2: Project Code
**File:** `database.py`
**Content:** `def query_users(): ... class User: ...`
**Result:** → `Project_DataApp/Code/database.py`

### Example 3: Meeting Minutes
**File:** `team_standup_2025_01_15.md`
**Content:** "Attendees:", "Action items:", "Next steps:"
**Result:** → `Meeting_Records/Notes/team_standup_2025_01_15.md`

### Example 4: Data Analysis
**File:** `sales_report_q1.xlsx`
**Content:** Time-series data, aggregations, formulas
**Result:** → `Data_Analysis/Analysis/sales_report_q1.xlsx`

---

## Customization

### Add More Project-Specific Subfolders
Edit [semantic/organization.py](semantic/organization.py) and update `CONTEXT_SUBFOLDERS`:

```python
CONTEXT_SUBFOLDERS = {
    'YourCustomContext': ['Subfolder1', 'Subfolder2', 'Subfolder3'],
}
```

### Adjust Detection Keywords
Edit [semantic/organization.py](semantic/organization.py) in the `_determine_subfolder()` method:

```python
subfolder_keywords = {
    'Your_Subfolder': ['keyword1', 'keyword2', 'file_extension'],
}
```

### Modify Context Detection
Edit `_is_job_search_file()` or `_detect_project()` methods to add custom detection logic.

---

## Benefits

✅ **Organized by Meaning** - Files grouped by how they're used, not just file type
✅ **Project Isolation** - Each project has its own folder with everything inside
✅ **Hierarchical** - Easy to navigate nested structures
✅ **Smart Distribution** - Similar files automatically sorted into appropriate subfolders
✅ **Scalable** - Grows naturally as you add more files and projects
✅ **Relationship-Aware** - Related files stay together

---

## What This Solves

❌ **Before:** All PDFs in one folder, all Word docs in another, no relationship awareness
✅ **After:** Job search docs grouped together, project files isolated, meeting notes organized

❌ **Before:** Hundreds of files in Documents folder with vague names
✅ **After:** Job_Search/Resume_Files/resume.pdf, Project_Alpha/Code/main.py, etc.

❌ **Before:** Hard to find related files across file types
✅ **After:** Everything for a project in one place with organized subfolders

---

## Running the Organizer

1. Open `main.py` (GUI will launch)
2. Select source directory (your unorganized files)
3. Select destination directory (where organized files go)
4. Click "Start Organization"
5. Watch as files are analyzed and organized into hierarchical structures
6. View the final folder structure in the log output

The system will automatically:
- Detect job search related files
- Identify project files and group them
- Create appropriate subfolders
- Sort files into the most relevant subfolders
- Display the complete hierarchical structure when done

**Result:** A clean, organized folder structure where everything has its place! 📁
