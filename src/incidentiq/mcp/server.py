
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
import csv


# Register the MCP server.
mcp = FastMCP("IncidentIQ")

#This is how the server know who the user is. The client sets this in the environment.
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
    #model_validate() is just a validation of the user context, not authentication. The authentication check is done in the require_access function.
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

    #reads each line, parses the JSON, and adds the resulting Python dictionary to a list.
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

#Business logic: find the incident and enforce RBAC.
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

#MCP interface: receive tool arguments and call the business logic
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


# ---------------------------------------------------------
# TICKET AND ASSET MCP TOOLS
# ---------------------------------------------------------

from pathlib import Path


def load_ticket_records() -> list[dict]:
    """
    Load tickets from the existing operational JSONL file.

    This reuses our synthetic ticket dataset.
    """
    path = (
        settings.incidentiq_data_dir
        / "operational"
        / "tickets.jsonl"
    )

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line.rstrip(","))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid ticket JSONL at line {line_number}"
                ) from exc

            records.append(record)

    return records


def require_it_staff(user: UserContext) -> None:
    """
    Restrict operational ticket searches to IT staff.

    Employee ticket access will be implemented separately
    with ownership-based filtering.
    """
    if user.role not in {"it_support", "it_admin"}:
        raise AccessDeniedError(
            "You do not have permission to access this operation."
        )


def ticket_summary(ticket: dict) -> dict:
    """
    Return only approved ticket fields.

    Explicit field selection prevents accidentally exposing
    new sensitive fields added to the dataset later.
    """
    allowed_fields = {
        "ticket_id",
        "title",
        "description",
        "status",
        "priority",
        "service_id",
        "requester_id",
        "assigned_team",
        "created_at",
        "updated_at",
    }

    return {
        key: value
        for key, value in ticket.items()
        if key in allowed_fields
    }


@mcp.tool(name="search_tickets")
def search_tickets_tool(
    query: str,
    limit: int = 5,
) -> dict:
    """
    Search existing tickets by keyword.

    Only IT Support and IT Admin may search all tickets.
    """
    user = get_authenticated_user()
    require_it_staff(user)

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20.")

    terms = query.lower().split()
    matches = []

    for ticket in load_ticket_records():
        require_access(
            user,
            ticket.get("classification", "it_internal"),
        )

        searchable = " ".join(
            str(ticket.get(field, ""))
            for field in (
                "ticket_id",
                "title",
                "description",
                "service_id",
                "status",
            )
        ).lower()

        if all(term in searchable for term in terms):
            matches.append(ticket_summary(ticket))

        if len(matches) >= limit:
            break

    return {
        "count": len(matches),
        "tickets": matches,
    }


@mcp.tool(name="get_ticket")
def get_ticket_tool(ticket_id: str) -> dict:
    """
    Retrieve one authorized ticket.

    Uses the same RBAC policy as ticket search.
    """
    user = get_authenticated_user()
    require_it_staff(user)

    if not re.fullmatch(r"TKT-\d{4}", ticket_id):
        raise ValueError("Invalid ticket ID format.")

    for ticket in load_ticket_records():
        if ticket.get("ticket_id") != ticket_id:
            continue

        require_access(
            user,
            ticket.get("classification", "it_internal"),
        )

        return {
            "found": True,
            "ticket": ticket_summary(ticket),
        }

    return {
        "found": False,
        "message": "Ticket not found.",
    }


def load_asset_records() -> list[dict]:
    """
    Load company asset inventory from CSV.

    csv.DictReader converts each row into a dictionary
    using the CSV column names as keys.
    """
    path = (
        settings.incidentiq_data_dir
        / "operational"
        / "assets.csv"
    )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def asset_summary(asset: dict) -> dict:
    """
    Return a limited asset projection.

    Only fields approved for the operational lookup
    are included in tool results.
    """
    allowed_fields = {
        "asset_id",
        "asset_type",
        "asset_name",
        "status",
        "service_id",
        "assigned_user_id",
    }

    return {
        key: value
        for key, value in asset.items()
        if key in allowed_fields
    }


@mcp.tool(name="search_assets")
def search_assets_tool(
    query: str,
    limit: int = 5,
) -> dict:
    """
    Search authorized IT asset records.

    Access is checked for each asset before returning it.
    """
    user = get_authenticated_user()
    require_it_staff(user)

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20.")

    terms = query.lower().split()
    matches = []

    for asset in load_asset_records():
        classification = asset.get(
            "classification",
            "it_internal",
        )

        try:
            require_access(user, classification)
        except AccessDeniedError:
            continue

        searchable = " ".join(
            str(value)
            for key, value in asset.items()
            if key in {
                "asset_id",
                "asset_type",
                "asset_name",
                "service_id",
                "status",
            }
        ).lower()

        if all(term in searchable for term in terms):
            matches.append(asset_summary(asset))

        if len(matches) >= limit:
            break

    return {
        "count": len(matches),
        "assets": matches,
    }


#When we start the server, we use this main() function to run it. The MCP protocol messages travel through stdout, so avoid using print() for ordinary debugging here. The server is started using the stdio transport, which means that it will communicate with clients through standard input and output streams. This is useful for running the server as a subprocess and allows for easy integration with other tools and scripts.
def main() -> None:
    """
    Start the MCP server using stdio.

    MCP protocol messages travel through stdout, so
    avoid using print() for ordinary debugging here.
    """

    mcp.run(transport="stdio") #starts the MCP server and tells it to communicate using standard input and output.


if __name__ == "__main__":
    main()
