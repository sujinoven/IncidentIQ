
import json
from collections import Counter
from pathlib import Path

from evaluation_models import EvaluationCase

DATA = Path("data")
EVAL_DIR = DATA / "evaluation"

documents = json.loads(
    (DATA / "metadata/documents.json").read_text(
        encoding="utf-8"
    )
)

valid_document_ids = {
    document["document_id"]
    for document in documents
}

all_cases = []
seen_ids = set()

for split in ("dev", "test"):
    path = EVAL_DIR / f"{split}.jsonl"

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        case = EvaluationCase.model_validate(
            json.loads(line)
        )

        if case.case_id in seen_ids:
            raise ValueError(
                f"Duplicate evaluation ID: {case.case_id}"
            )

        seen_ids.add(case.case_id)

        if case.split != split:
            raise ValueError(
                f"Incorrect split: {case.case_id}"
            )

        for doc_id in case.expected_document_ids:
            if doc_id not in valid_document_ids:
                raise ValueError(
                    f"Unknown document: {doc_id}"
                )

        all_cases.append(case)

assert len(all_cases) == 60

category_counts = Counter(
    case.category for case in all_cases
)

expected_counts = {
    "knowledge_qa": 20,
    "incident_analysis": 15,
    "mcp_operations": 10,
    "security": 10,
    "failure_handling": 5,
}

assert dict(category_counts) == expected_counts

split_counts = Counter(case.split for case in all_cases)

assert split_counts["dev"] == 40
assert split_counts["test"] == 20

print("Validated 60 evaluation cases")
print("Categories:", dict(category_counts))
print("Splits:", dict(split_counts))
