
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from incident_models import Incident
from ticket_models import TicketRecord
from asset_models import Asset
from evaluation_models import EvaluationCase


ROOT = Path("data")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    records = []

    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{number}: invalid JSON"
                ) from exc

    return records


def unique_ids(records, key):
    ids = [record[key] for record in records]

    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate values for {key}")

    return set(ids)


def require_count(label, actual, expected):
    if actual != expected:
        raise ValueError(
            f"{label}: expected {expected}, found {actual}"
        )
    print(f"PASS {label}: {actual}")


class ServiceCheck(BaseModel):
    service_id: str = Field(pattern=r"^SVC-[A-Z]+$")
    service_name: str
    description: str
    owner_team: str
    status: str
    checked_at: datetime
    active_alerts: list[str]


# -------------------------------------------------
# 1. Services and users
# -------------------------------------------------

services = [
    ServiceCheck.model_validate(row).model_dump(mode="json")
    for row in load_json(ROOT / "operational/services.json")
]

users = load_json(ROOT / "operational/users.json")

require_count("services", len(services), 10)
require_count("users", len(users), 12)

service_ids = unique_ids(services, "service_id")
user_ids = unique_ids(users, "user_id")

user_by_id = {user["user_id"]: user for user in users}

# -------------------------------------------------
# 2. Knowledge documents
# -------------------------------------------------

documents = load_json(ROOT / "metadata/documents.json")

require_count("knowledge documents", len(documents), 35)

document_ids = unique_ids(documents, "document_id")
document_by_id = {
    document["document_id"]: document
    for document in documents
}

allowed_classifications = {
    "employee_public",
    "it_internal",
    "admin_restricted",
}

allowed_types = {"policy", "runbook", "kb_article"}

for document in documents:
    if document["service_id"] not in service_ids:
        raise ValueError(
            f"Unknown document service: {document['document_id']}"
        )

    if document["classification"] not in allowed_classifications:
        raise ValueError(
            f"Invalid classification: {document['document_id']}"
        )

    if document["document_type"] not in allowed_types:
        raise ValueError(
            f"Invalid document type: {document['document_id']}"
        )

    content_path = (
        ROOT / document["content_file"]
    ).resolve()

    # Prevent metadata paths from escaping the data directory.
    if not content_path.is_relative_to(ROOT.resolve()):
        raise ValueError("Document path escapes data directory")

    if not content_path.is_file():
        raise FileNotFoundError(content_path)

    if not content_path.read_text(
        encoding="utf-8"
    ).strip():
        raise ValueError(f"Empty document: {content_path}")

type_counts = Counter(
    document["document_type"] for document in documents
)

assert type_counts["policy"] == 8
assert type_counts["runbook"] == 12
assert type_counts["kb_article"] == 15

print("PASS knowledge document references")

# -------------------------------------------------
# 3. Historical incidents
# -------------------------------------------------

incidents = [
    Incident.model_validate(row)
    for row in load_jsonl(
        ROOT / "operational/incidents.jsonl"
    )
]

require_count("incidents", len(incidents), 30)

unique_ids(
    [incident.model_dump() for incident in incidents],
    "incident_id",
)

for incident in incidents:
    if incident.service_id not in service_ids:
        raise ValueError(
            f"Unknown incident service: {incident.incident_id}"
        )

    for runbook_id in incident.runbook_ids:
        document = document_by_id.get(runbook_id)

        if document is None:
            raise ValueError(
                f"Missing runbook: {runbook_id}"
            )

        if document["document_type"] != "runbook":
            raise ValueError(
                f"Not a runbook: {runbook_id}"
            )

print("PASS incident references")

# -------------------------------------------------
# 4. Historical tickets
# -------------------------------------------------

tickets = [
    TicketRecord.model_validate(row)
    for row in load_jsonl(
        ROOT / "operational/tickets.jsonl"
    )
]

require_count("tickets", len(tickets), 25)

unique_ids(
    [ticket.model_dump() for ticket in tickets],
    "ticket_id",
)

for ticket in tickets:
    if ticket.service_id not in service_ids:
        raise ValueError(
            f"Unknown ticket service: {ticket.ticket_id}"
        )

    if ticket.requester_id not in user_ids:
        raise ValueError(
            f"Unknown ticket requester: {ticket.ticket_id}"
        )

    if not user_by_id[ticket.requester_id]["active"]:
        raise ValueError(
            f"Inactive requester: {ticket.ticket_id}"
        )

print("PASS ticket references")

# -------------------------------------------------
# 5. Assets
# -------------------------------------------------

with (ROOT / "operational/assets.csv").open(
    newline="", encoding="utf-8"
) as file:
    rows = list(csv.DictReader(file))

assets = []

for row in rows:
    row["assigned_to"] = row["assigned_to"] or None
    assets.append(Asset.model_validate(row))

require_count("assets", len(assets), 15)

unique_ids(
    [asset.model_dump() for asset in assets],
    "asset_id",
)

for asset in assets:
    if asset.service_id not in service_ids:
        raise ValueError(
            f"Unknown asset service: {asset.asset_id}"
        )

    if asset.assigned_to and asset.assigned_to not in user_ids:
        raise ValueError(
            f"Unknown asset owner: {asset.asset_id}"
        )

print("PASS asset references")

# -------------------------------------------------
# 6. Evaluation dataset
# -------------------------------------------------

cases = []

for split in ("dev", "test"):
    path = ROOT / "evaluation" / f"{split}.jsonl"

    for row in load_jsonl(path):
        case = EvaluationCase.model_validate(row)

        if case.split != split:
            raise ValueError(
                f"Incorrect split: {case.case_id}"
            )

        for document_id in case.expected_document_ids:
            if document_id not in document_ids:
                raise ValueError(
                    f"Unknown evaluation document: {document_id}"
                )

        cases.append(case)

require_count("evaluation cases", len(cases), 60)

unique_ids(
    [case.model_dump() for case in cases],
    "case_id",
)

splits = Counter(case.split for case in cases)

assert splits["dev"] == 40
assert splits["test"] == 20

categories = Counter(case.category for case in cases)

assert categories == {
    "knowledge_qa": 20,
    "incident_analysis": 15,
    "mcp_operations": 10,
    "security": 10,
    "failure_handling": 5,
}

print("PASS evaluation coverage")

print()
print("=" * 48)
print("INCIDENTIQ DATASET VALIDATION PASSED")
print("=" * 48)
