FILE_TYPE_CATEGORIES = {
    ".pdf": "Report",
    ".docx": "Report",
    ".csv": "Spreadsheet",
    ".py": "Code",
    ".pptx": "Presentation",
    ".jpg": "Image",
}

def get_file_category_by_extension(path):
    import os
    return FILE_TYPE_CATEGORIES.get(os.path.splitext(path)[1].lower())
