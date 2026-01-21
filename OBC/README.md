# OBC (Organized By Classification)

OBC is a Python-based **content-first file organization system** with a **Tkinter GUI**. It extracts text from files (including OCR for images/scanned documents), classifies the content into themes, and then organizes files into folders based on meaning rather than extension.

## Why this project (portfolio framing)

This project demonstrates:

- Building a practical **automation tool** with a GUI
- A real pipeline: **scan → extract → classify → organize**
- Combining classical NLP + ML concepts with pragmatic fallbacks
- Robustness work: path validation, logging, safe file moves, and error handling

## Key Features

- **GUI-driven workflow** (Tkinter)
- **Multi-format text extraction**
  - PDF (PyMuPDF)
  - DOCX (python-docx)
  - TXT
  - Images (OCR via Tesseract if available)
- **Classification**
  - File-type high-confidence shortcuts for known formats
  - Context-aware classification (when enabled)
  - Zero-shot classification via Transformers (BART MNLI) when available
  - Deterministic fallback classification when the model isn’t available
- **Organization**
  - Theme-based organization
  - Optional hierarchical structure using semantic relationships
- **Logging**
  - Rotating file logs to `logs/file_organizer.log`
  - Console logs for live feedback

## How it works (pipeline)

The core flow (`core/pipeline.py`) is effectively:

1. **Validate paths** (source exists/readable, destination is valid)
2. **Collect files** recursively
3. **First pass: analyze**
   - Extract text (`utils/extraction.py`)
   - Classify into a theme + confidence (`classification/classifier.py`)
4. **Second pass: build folder structure**
   - If enabled and multiple docs exist: generate a hierarchical structure based on relationships (`semantic/organization.py`)
   - Otherwise: simple theme-based folders
5. **Third pass: move files**
   - Safe move with conflict handling (`core/organizer.py`)
6. **Cleanup** empty source subfolders (best effort)

## Requirements

- **Python** 3.10+
- **Tesseract OCR** (recommended for OCR features)
  - On Windows, `main.py` tries common install locations automatically (e.g. `C:\Program Files\Tesseract-OCR\tesseract.exe`)

## Install

```bash
pip install -r requirements.txt
```

## Run

Launch the GUI:

```bash
python main.py
```

## Configuration

Configuration is JSON-driven:

- `config/config.json` — classification/labeling rules
- `config/gui_config.json` — GUI defaults
- `config/tesseract_config.json` — OCR settings

## Outputs

- **Logs**: `logs/file_organizer.log`
- **Evaluation artifacts** (already included in this repo): `evaluation_reports/` (CSVs + plots)

## Repo Structure (high level)

- `main.py` — entrypoint (logging + NLTK + Tesseract setup + GUI)
- `file_organizer_gui.py` — Tkinter GUI
- `core/` — pipeline, scanner, organizer, dynamic label helpers
- `classification/` — classifier + fallbacks + file-type logic
- `semantic/` — relationship-aware organization and label utilities
- `evaluation/` + `evaluation_reports/` — evaluation tooling + generated reports

## Troubleshooting

- **OCR not working**
  - Install Tesseract and ensure it’s discoverable (PATH or standard install path).
  - If Tesseract is missing, OCR returns empty text and those files fall back to non-OCR logic.
- **Slow first run**
  - Transformers + Torch can be heavy; OBC uses fallbacks if the model can’t be loaded.


## Demo in 60 seconds

1. Launch the GUI: `python main.py`
2. Select a source folder containing mixed PDFs/DOCX/images
3. Select an output folder
4. Run organization and show the resulting themed/hierarchical folders
5. Open `logs/file_organizer.log` to show traceability

## Screenshots

Add screenshots/GIFs to: `docs/screenshots/`

- `docs/screenshots/gui.png`
- `docs/screenshots/output_structure.png`