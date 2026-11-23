
"""
IncidentIQ MCP Server

Exposes read-only operational tools for NovaTech Solutions.

Tools:
    get_incident
    search_incidents
    get_service_status

Security:
    - User identity comes from trusted process configuration.
    - Every incident is checked against RBAC permissions.
    - Restricted data is not returned to unauthorized users.

Transport:
    stdio
"""

import json
import os
import re
from typing import Any

from mcp.server.fastmcp import FastMCP

from incidentiq.core.config import settings
from incidentiq.core.models import UserContext
from incidentiq.core.security import (
    AccessDeniedError,
    require_access,
)


# Register the MCP server.
mcp = FastMCP("IncidentIQ")


def get_authenticated_user() -> UserContext:
    """
    Obtain the user context assigned by the host application.

    The LLM cannot pass a role as an MCP tool argument.

    Important:
        This environment-based identity is appropriate for
        a local, isolated demo subprocess. It is not a
        production authentication mechanism.
    """

    user_id = os.environ.get("INCIDENTIQ_USER_ID")
    role = os.environ.get("INCIDENTIQ_USER_ROLE")

    if not user_id or not role:
        raise PermissionError(
            "Authenticated MCP user context is missing."
        )

    return UserContext.model_validate(
        {
            "user_id": user_id,
            "role": role,
        }
    )


def load_jsonl_records(filename: str) -> list[dict[str, Any]]:
    """
    Load operational JSONL records.

    The dataset currently tolerates trailing commas
    from the original synthetic data generation.

    Later we'll normalize the files to strict JSONL.
    """

    allowed_files = {"incidents.jsonl"}

    if filename not in allowed_files:
        raise ValueError("Unsupported operational file.")

    path = (
        settings.incidentiq_data_dir
        / "operational"
        / filename
    )

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line.rstrip(","))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL record at line {line_number}"
                ) from exc

            records.append(record)

    return records


def get_incident(
    incident_id: str,
    user: UserContext,
) -> dict | None:
    """
    Shared Python service for retrieving an incident.

    Why:
        LangGraph can reuse this service directly while
        MCP exposes the same operation as a tool.

    Security:
        Access is checked before incident details return.
    """

    if not re.fullmatch(r"INC-\d{4}", incident_id):
        raise ValueError("Invalid incident ID format.")

    for incident in load_jsonl_records("incidents.jsonl"):
        if incident["incident_id"] != incident_id:
            continue

        require_access(
            user,
            incident["classification"],
        )

        return incident

    return None


@mcp.tool(name="get_incident")
def get_incident_tool(incident_id: str) -> dict:
    """
    MCP tool: Retrieve one authorized incident.

    The model supplies only the incident ID.
    The server obtains the user context independently.
    """

    user = get_authenticated_user()

    try:
        incident = get_incident(incident_id, user)
    except AccessDeniedError:
        incident = None

    if incident is None:
        return {
            "found": False,
            "message": (
                "Incident not found or access denied."
            ),
        }

    return {
        "found": True,
        "incident": incident,
    }


@mcp.tool(name="search_incidents")
def search_incidents_tool(
    query: str,
    limit: int = 5,
) -> dict:
    """
    MCP tool: Search authorized historical incidents.

    This initial implementation performs keyword matching
    against incident titles, symptoms, and service IDs.
    """

    user = get_authenticated_user()

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20.")

    terms = query.lower().split()
    matches = []

    for incident in load_jsonl_records("incidents.jsonl"):
        try:
            require_access(
                user,
                incident["classification"],
            )
        except AccessDeniedError:
            continue

        searchable_text = " ".join(
            [
                incident.get("title", ""),
                incident.get("service_id", ""),
                " ".join(incident.get("symptoms", [])),
            ]
        ).lower()

        if all(term in searchable_text for term in terms):
            matches.append(incident)

        if len(matches) >= limit:
            break

    return {
        "count": len(matches),
        "incidents": matches,
    }


@mcp.tool(name="get_service_status")
def get_service_status_tool(service_id: str) -> dict:
    """
    MCP tool: Retrieve a service's recorded health status.

    This uses the synthetic service snapshot, not live
    production monitoring data.
    """

    # Require a valid application user even for this tool.
    get_authenticated_user()

    if not re.fullmatch(r"SVC-[A-Z0-9-]+", service_id):
        raise ValueError("Invalid service ID format.")

    path = (
        settings.incidentiq_data_dir
        / "operational"
        / "services.json"
    )

    services = json.loads(
        path.read_text(encoding="utf-8")
    )

    for service in services:
        if service["service_id"] == service_id:
            return {
                "found": True,
                "service": service,
            }

    return {
        "found": False,
        "message": "Service not found.",
    }


def main() -> None:
    """
    Start the MCP server using stdio.

    MCP protocol messages travel through stdout, so
    avoid using print() for ordinary debugging here.
    """

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
