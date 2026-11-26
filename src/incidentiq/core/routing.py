#Creating a shared request router


import re

from typing import Literal


# =========================================================
# 1. SUPPORTED REQUEST ROUTES
# =========================================================


RequestRoute = Literal[
    "knowledge",
    "incident",
    "ticket",
    "get_service_status",
    "search_incidents",
    "search_tickets",
    "get_ticket",
    "search_assets",
]


OPERATIONAL_ROUTES = frozenset({
    "get_service_status",
    "search_incidents",
    "search_tickets",
    "get_ticket",
    "search_assets",
})


STAFF_ONLY_ROUTES = frozenset({
    "search_tickets",
    "get_ticket",
    "search_assets",
})


# =========================================================
# 2. IDENTIFIER PATTERNS
# =========================================================


INCIDENT_ID_PATTERN = (
    r"\bINC-\d{4}\b"
)


SERVICE_ID_PATTERN = (
    r"\bSVC-[A-Z0-9-]+\b"
)


TICKET_ID_PATTERN = (
    r"\bTKT-(?:\d{4}|[A-F0-9]{12})\b"
)


# =========================================================
# 3. SHARED REQUEST CLASSIFICATION
# =========================================================


def classify_request(
    query: str,
) -> RequestRoute:

    text = " ".join(
        query.lower().split()
    )

    # -----------------------------------------------------
    # Ticket preparation
    # -----------------------------------------------------

    if re.match(

        r"^(?:please\s+)?"
        r"(?:create|open|raise|submit|prepare)\s+"
        r"(?:(?:a|an|new|support|it)\s+)*ticket\b",

        text,
    ):

        return "ticket"

    # -----------------------------------------------------
    # Specific ticket lookup
    # -----------------------------------------------------

    if re.search(
        TICKET_ID_PATTERN,
        query.upper(),
    ):

        return "get_ticket"

    # -----------------------------------------------------
    # Ticket search
    # -----------------------------------------------------

    if re.search(

        r"\b(search|find|list|show)\b"
        r".*\btickets?\b",

        text,
    ):

        return "search_tickets"

    # -----------------------------------------------------
    # Asset search
    # -----------------------------------------------------

    if re.search(

        r"\b(search|find|list|show)\b"
        r".*\bassets?\b",

        text,
    ):

        return "search_assets"

    # -----------------------------------------------------
    # Specific incident investigation
    # -----------------------------------------------------

    if re.search(
        INCIDENT_ID_PATTERN,
        query.upper(),
    ):

        return "incident"

    # -----------------------------------------------------
    # Incident search
    # -----------------------------------------------------

    if re.search(

        r"\b(search|find|list|show)\b"
        r".*\bincidents?\b",

        text,
    ):

        return "search_incidents"

    # -----------------------------------------------------
    # Service health
    # -----------------------------------------------------

    if (
        re.search(
            SERVICE_ID_PATTERN,
            query.upper(),
        )

        and re.search(

            r"\b(status|health|available|availability)\b",

            text,
        )
    ):

        return "get_service_status"

    # -----------------------------------------------------
    # Incident investigation without an ID
    # -----------------------------------------------------

    if re.search(

        r"\b(investigate|analyze)\b"
        r".*\bincident\b",

        text,
    ):

        return "incident"

    # -----------------------------------------------------
    # Default: enterprise knowledge QA
    # -----------------------------------------------------

    return "knowledge"


# =========================================================
# 4. IDENTIFIER EXTRACTION
# =========================================================


def extract_single_identifier(
    query: str,
    pattern: str,
) -> str | None:

    identifiers = set(
        re.findall(
            pattern,
            query.upper(),
        )
    )

    if len(identifiers) != 1:

        return None

    return next(
        iter(identifiers)
    )


# =========================================================
# 5. OPERATIONAL SEARCH TERM EXTRACTION
# =========================================================


def extract_search_terms(
    query: str,
    route: RequestRoute,
) -> str:

    text = query.strip()

    text = re.sub(

        r"^(?:please\s+)?"
        r"(?:search|find|list|show)\s+",

        "",

        text,

        flags=re.IGNORECASE,
    )

    text = re.sub(

        r"^(?:me|all|the|for)\s+",

        "",

        text,

        flags=re.IGNORECASE,
    )

    entity_patterns = {

        "search_incidents": r"\bincidents?\b",

        "search_tickets": r"\btickets?\b",

        "search_assets": r"\bassets?\b",
    }

    pattern = entity_patterns.get(
        route
    )

    if pattern:

        text = re.sub(
            pattern,
            " ",
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(

        r"\b(for|related to|about|matching)\b",

        " ",

        text,

        flags=re.IGNORECASE,
    )

    return " ".join(
        text.strip(" ?.,:").split()
    )


# =========================================================
# 6. MODULE SMOKE TEST
# =========================================================


if __name__ == "__main__":

    queries = [

        "What does VPN Error 809 mean?",

        "Investigate incident INC-0012",

        "Check status of SVC-VPN",

        "Search VPN incidents",

        "Search VPN tickets",

        "Show TKT-0001",

        "Find laptop assets",

        "Create ticket",

    ]

    for query in queries:

        route = classify_request(query)

        print(
            f"{query} => {route}"
        )
