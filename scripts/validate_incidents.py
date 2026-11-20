
import json
from pathlib import Path

from incident_models import Incident

data_root = Path("data")

services = json.loads(
    (data_root / "operational/services.json").read_text(
        encoding="utf-8"
    )
)

documents = json.loads(
    (data_root / "metadata/documents.json").read_text(
        encoding="utf-8"
    )
)

valid_services = {s["service_id"] for s in services}

valid_runbooks = {
    doc["document_id"]
    for doc in documents
    if doc["document_type"] == "runbook"
}

path = data_root / "operational/incidents.jsonl"

incident_ids = set()
count = 0

for line_number, line in enumerate(
    path.read_text(encoding="utf-8").splitlines(),
    start=1,
):
    if not line.strip():
        continue

    incident = Incident.model_validate(json.loads(line))

    if incident.incident_id in incident_ids:
        raise ValueError(
            f"Duplicate incident: {incident.incident_id}"
        )

    incident_ids.add(incident.incident_id)

    if incident.service_id not in valid_services:
        raise ValueError(
            f"Unknown service: {incident.service_id}"
        )

    for runbook_id in incident.runbook_ids:
        if runbook_id not in valid_runbooks:
            raise ValueError(
                f"Unknown runbook: {runbook_id}"
            )

    count += 1

print(f"Validated {count} incidents")
