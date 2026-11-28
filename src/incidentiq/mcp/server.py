
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
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from pathlib import Path

from incidentiq.core.security import (
    ROLE_PERMISSIONS,
)

from pathlib import Path

from incidentiq.core.security import (
    ROLE_PERMISSIONS,
)

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

# =========================================================
# OPERATIONAL INCIDENT SEARCH NORMALIZATION
# =========================================================

INCIDENT_SEARCH_ALIASES = {
    "problems": "failure",
    "problem": "failure",
    "errors": "failure",
    "error": "failure",
    "failures": "failure",
    "failed": "failure",
    "fails": "failure",
    "outages": "outage",
    "dashboards": "dashboard",
    "connections": "connection",
    "disconnections": "disconnection",
    "folders": "folder",
    "shared": "sharing",
    "files": "file",
    "unavailable": "outage",
}

INCIDENT_SEARCH_STOP_WORDS = frozenset({
    "a",
    "an",
    "the",
    "for",
    "of",
    "to",
    "with",
    "about",
    "related",
    "historical",
    "incident",
    "incidents",
    "corporate",
})


def normalize_incident_search(
    text: str,
) -> set[str]:
    """
    Convert incident search text into normalized tokens.

    This is deterministic lexical normalization,
    not semantic vector retrieval.
    """

    tokens = re.findall(
        r"[a-z0-9]+",
        text.casefold(),
    )

    normalized = {
        INCIDENT_SEARCH_ALIASES.get(token, token)
        for token in tokens
        if token not in INCIDENT_SEARCH_STOP_WORDS
    }

    return normalized

@mcp.tool(name="search_incidents")
def search_incidents_tool(
    query: str,
    limit: int = 5,
) -> dict:
    """
    Search authorized historical incidents using
    normalized lexical matching.

    Authorization is enforced before matching
    or ranking incident records.
    """

    user = get_authenticated_user()

    if not query.strip():
        raise ValueError(
            "Search query cannot be empty."
        )

    if len(query) > 500:
        raise ValueError(
            "Search query is too long."
        )

    if not 1 <= limit <= 20:
        raise ValueError(
            "limit must be between 1 and 20."
        )

    query_terms = normalize_incident_search(
        query
    )

    if not query_terms:
        return {
            "count": 0,
            "incidents": [],
        }

    ranked_matches = []

    for incident in load_jsonl_records(
        "incidents.jsonl"
    ):

        # Authorization must happen first.

        try:
            require_access(
                user,
                incident["classification"],
            )

        except AccessDeniedError:
            continue

        searchable_text = " ".join([
            incident.get("title", ""),
            incident.get("service_id", ""),
            " ".join(
                incident.get("symptoms", [])
            ),
        ])

        record_terms = normalize_incident_search(
            searchable_text
        )

        matched_terms = (
            query_terms & record_terms
        )

        # Require all normalized search terms.
        # Do not return weak partial matches.

        if not query_terms.issubset(
            record_terms
        ):
            continue

        title_terms = normalize_incident_search(
            incident.get("title", "")
        )

        title_matches = len(
            query_terms & title_terms
        )

        ranked_matches.append((
            title_matches,
            len(matched_terms),
            incident,
        ))

    # Prefer records whose titles directly
    # match more of the requested search terms.

    ranked_matches.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            item[2]["incident_id"],
        )
    )

    matches = [
        item[2]
        for item in ranked_matches[:limit]
    ]

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
    Load historical tickets from JSONL and newly
    created tickets from SQLite.
    """
    path = (
        settings.incidentiq_data_dir
        / "operational"
        / "tickets.jsonl"
    )

    tickets = []

    # Load historical synthetic tickets.
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                tickets.append(json.loads(line.rstrip(",")))

    # Load newly created tickets.
    with get_ticket_database() as db:
        rows = db.execute(
            "SELECT ticket_json FROM created_tickets"
        ).fetchall()

    for row in rows:
        tickets.append(json.loads(row["ticket_json"]))

    return tickets



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

def access_denied_result() -> dict:
    """
    Return a safe authorization denial.

    Do not disclose resource existence,
    classifications or internal records.
    """

    return {
        "success": False,
        "error": "ACCESS_DENIED",
        "message": (
            "You do not have permission "
            "to access this operation."
        ),
    }

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

    try:
        require_it_staff(user)

    except AccessDeniedError:
        return access_denied_result()

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20.")

    terms = query.lower().split()
    matches = []

    for ticket in load_ticket_records():
        try:
            require_access(
                user,
                ticket.get("classification", "it_internal"),
            )
        except AccessDeniedError:
            continue

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

    try:
        require_it_staff(user)

    except AccessDeniedError:
        return access_denied_result()

    if not re.fullmatch(r"TKT-(?:\d{4}|[A-F0-9]{12})",ticket_id,):
        raise ValueError("Invalid ticket ID format.")

    for ticket in load_ticket_records():
        if ticket.get("ticket_id") != ticket_id:
            continue

        try:
                require_access(
                    user,
                    ticket.get(
                        "classification",
                        "it_internal",
                    ),
                )

        except AccessDeniedError:
                return {
                    "found": False,
                    "message": (
                        "Ticket not found or access denied."
                    ),
                }

        return {
            "found": True,
            "ticket": ticket_summary(ticket),
        }

    return {
    "found": False,
    "message": (
        "Ticket not found or access denied."
    ),
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

    try:
        require_it_staff(user)

    except AccessDeniedError:
        return access_denied_result()

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


# ---------------------------------------------------------
# SAFE TICKET CREATION
# ---------------------------------------------------------

from pydantic import BaseModel, Field


class NewTicket(BaseModel):
    """
    Validate the information needed to create a ticket.

    The requester ID is deliberately excluded.
    It comes from the trusted application context.
    """

    service_id: str = Field(
        pattern=r"^SVC-[A-Z0-9-]+$"
    )
    title: str = Field(min_length=5, max_length=150)
    description: str = Field(min_length=10, max_length=2000)
    priority: str = Field(pattern=r"^P[1-4]$")


def get_ticket_database() -> sqlite3.Connection:
    """
    Open SQLite storage for pending and newly created tickets.

    SQLite transactions let us consume a confirmation token
    and create its ticket as one atomic operation.
    """
    path = settings.incidentiq_sqlite_path
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        str(path),
        timeout=10,
    )
    connection.row_factory = sqlite3.Row

    connection.execute("""
        CREATE TABLE IF NOT EXISTS pending_tickets (
            token TEXT PRIMARY KEY,
            requester_id TEXT NOT NULL,
            ticket_json TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            created_ticket_id TEXT
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS created_tickets (
            ticket_id TEXT PRIMARY KEY,
            requester_id TEXT NOT NULL,
            ticket_json TEXT NOT NULL
        )
    """)

    connection.commit()
    return connection


def service_exists(service_id: str) -> bool:
    """
    Check that the requested service exists in our dataset.
    """
    path = (
        settings.incidentiq_data_dir
        / "operational"
        / "services.json"
    )

    services = json.loads(path.read_text(encoding="utf-8"))

    return any(
        service["service_id"] == service_id
        for service in services
    )


@mcp.tool(name="prepare_ticket")
def prepare_ticket_tool(
    service_id: str,
    title: str,
    description: str,
    priority: str = "P3",
) -> dict:
    """
    Validate and prepare a ticket without creating it.

    Returns a confirmation token that the trusted UI must
    retain until the user explicitly approves the request.
    """
    user = get_authenticated_user()

    ticket = NewTicket(
        service_id=service_id,
        title=title,
        description=description,
        priority=priority,
    )

    if not service_exists(ticket.service_id):
        raise ValueError("Unknown service ID.")

    token = secrets.token_urlsafe(32)
    expires_at = (
        datetime.now(timezone.utc) + timedelta(minutes=10)
    ).isoformat()

    preview = {
        **ticket.model_dump(),
        "requester_id": user.user_id,
    }

    with get_ticket_database() as db:
        db.execute(
            """
            INSERT INTO pending_tickets
            (token, requester_id, ticket_json, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                token,
                user.user_id,
                json.dumps(preview),
                expires_at,
            ),
        )

    return {
        "status": "awaiting_confirmation",
        "preview": preview,
        "confirmation_token": token,
        "expires_at": expires_at,
        "message": "Ticket prepared but not created.",
    }


@mcp.tool(name="create_ticket")
def create_ticket_tool(
    confirmation_token: str,
) -> dict:
    """
    Create a previously prepared ticket.

    IMPORTANT:
        This function must be invoked only by a trusted
        application handler after an explicit user
        confirmation event.

        The LLM must never be given unrestricted access
        to this tool.
    """
    user = get_authenticated_user()

    db = get_ticket_database()

    try:
        # Lock the database for this transaction.
        db.execute("BEGIN IMMEDIATE")

        pending = db.execute(
            """
            SELECT *
            FROM pending_tickets
            WHERE token = ?
            """,
            (confirmation_token,),
        ).fetchone()

        if pending is None:
            raise ValueError("Invalid confirmation token.")

        # A token is bound to its original requester.
        if pending["requester_id"] != user.user_id:
            raise AccessDeniedError(
                "Confirmation belongs to another user."
            )

        # A previously completed request is idempotent.
        if pending["created_ticket_id"]:
            ticket_id = pending["created_ticket_id"]
            db.commit()
            return {
                "status": "created",
                "ticket_id": ticket_id,
                "already_created": True,
            }

        expires_at = datetime.fromisoformat(
            pending["expires_at"]
        )

        if datetime.now(timezone.utc) > expires_at:
            raise ValueError("Confirmation token expired.")

        ticket = json.loads(pending["ticket_json"])

        # Revalidate the prepared fields.
        validated = NewTicket.model_validate(ticket)

        if not service_exists(validated.service_id):
            raise ValueError("Service no longer exists.")

        # UUID-based IDs avoid collisions across processes.
        ticket_id = f"TKT-{uuid4().hex[:12].upper()}"

        record = {
            "ticket_id": ticket_id,
            **validated.model_dump(),
            "requester_id": user.user_id,
            "status": "open",
            "classification": "it_internal",
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        db.execute(
            """
            INSERT INTO created_tickets
            (ticket_id, requester_id, ticket_json)
            VALUES (?, ?, ?)
            """,
            (
                ticket_id,
                user.user_id,
                json.dumps(record),
            ),
        )

        db.execute(
            """
            UPDATE pending_tickets
            SET created_ticket_id = ?
            WHERE token = ?
            """,
            (ticket_id, confirmation_token),
        )

        db.commit()

        return {
            "status": "created",
            "ticket_id": ticket_id,
            "ticket": record,
            "already_created": False,
        }

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


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
