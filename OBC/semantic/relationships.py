from collections import defaultdict
import os

def detect_file_relationships(documents):
    relations = defaultdict(list)

    for doc in documents:
        base = os.path.splitext(doc["filename"])[0]
        for other in documents:
            if other is not doc and base in other["filename"]:
                relations[doc["path"]].append(other["path"])

    return relations
