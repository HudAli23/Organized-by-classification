from collections import defaultdict
import os

def get_content_based_clusters(documents):
    clusters = defaultdict(list)

    for doc in documents:
        ext = os.path.splitext(doc["filename"])[1]
        key = "Code" if ext == ".py" else "Documents"
        clusters[key].append(doc)

    return dict(clusters)
