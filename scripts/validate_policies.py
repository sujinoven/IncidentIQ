
import json
from pathlib import Path

DATA_ROOT = Path("data")

metadata_path = DATA_ROOT / "metadata" / "documents.json"
documents = json.loads(metadata_path.read_text(encoding="utf-8"))

document_ids = set()

for document in documents:
    doc_id = document["document_id"]

    if doc_id in document_ids:
        raise ValueError(f"Duplicate document ID: {doc_id}")

    document_ids.add(doc_id)

    content_path = DATA_ROOT / document["content_file"]

    if not content_path.is_file():
        raise FileNotFoundError(
            f"Missing content file for {doc_id}: {content_path}"
        )

    content = content_path.read_text(encoding="utf-8")

    if not content.strip():
        raise ValueError(f"Empty document: {doc_id}")

    if document["classification"] not in {
        "employee_public",
        "it_internal",
        "admin_restricted",
    }:
        raise ValueError(
            f"Invalid classification for {doc_id}"
        )

print(f"Validated {len(documents)} document records")
