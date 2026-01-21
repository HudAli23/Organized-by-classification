import os

def collect_all_files(root):
    files = []
    for r, _, f in os.walk(root):
        for file in f:
            path = os.path.join(r, file)
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                files.append(path)
    return files
