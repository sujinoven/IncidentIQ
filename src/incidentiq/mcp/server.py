#Implementing read only incident logic for the langgraph agent. This is a temporary solution until we can implement a more robust incident handling system.

import json
from pathlib import Path

from incidentiq.core.config import settings
from incidentiq.core.models import (
    Classification,
    UserContext,
)
from incidentiq.core.security import require_access


def get_incident(
    incident_id: str,
    user: UserContext,
) -> dict | None:
    """
    Retrieve an incident only when the user
    has permission to access its classification.
    """

    incidents_path = (
        settings.incidentiq_data_dir
        / "operational"
        / "incidents.jsonl"
    )#Our incidents are stored in incidents.jsonl, where each line represents one JSON record.This lets us read records individually without loading the entire file into memory.

    with incidents_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            # Tolerates trailing commas in the
            # current synthetic dataset.
            record = json.loads(line.rstrip(","))

            if record["incident_id"] != incident_id:
                continue

            classification: Classification = (
                record["classification"]
            )

            require_access(user, classification) #Even if an incident exists, we shouldn't return it to an unauthorized user.

            return record

    return None  #If the incident ID doesn't exist, we return None so the workflow can handle the missing record gracefully.
