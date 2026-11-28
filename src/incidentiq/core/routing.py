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
# 2A. SERVICE NAME ALIASES
# =========================================================


SERVICE_ALIASES: dict[str, tuple[str, ...]] = {

    "SVC-VPN": (
        "vpn",
        "corporate vpn",
        "virtual private network",
    ),

    "SVC-IDENTITY": (
        "identity",
        "identity and sso",
        "identity sso",
        "sso",
        "single sign on",
        "mfa",
    ),

    "SVC-EMAIL": (
        "email",
        "corporate email",
        "email service",
        "mail service",
    ),

    "SVC-PAYROLL": (
        "payroll",
        "payroll application",
        "payroll portal",
        "payroll service",
    ),

    "SVC-HR": (
        "hr",
        "hr portal",
        "human resources portal",
    ),

    "SVC-ENDPOINT": (
        "endpoint",
        "endpoint service",
        "employee devices",
        "managed devices",
    ),

    "SVC-WIFI": (
        "wifi",
        "wi fi",
        "office wifi",
        "office wi fi",
        "wireless network",
    ),

    "SVC-FILES": (
        "file sharing",
        "file storage",
        "shared folders",
        "shared folder",
        "file sharing service",
    ),

    "SVC-DEVOPS": (
        "devops",
        "ci cd",
        "ci cd platform",
        "deployment pipeline",
        "build pipeline",
    ),

    "SVC-MONITOR": (
        "monitoring",
        "monitoring platform",
        "monitoring dashboard",
        "monitoring service",
    ),
}

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
    # Incident search and historical investigation
    # -----------------------------------------------------

    explicit_incident_search = bool(
        re.search(
            r"\b(search|find|list|show|compare)\b"
            r".*\bincidents?\b",
            text,
        )
    )

    historical_investigation = (
        bool(
            re.match(
                r"^(?:please\s+)?"
                r"(?:investigate|analyze|find|search|"
                r"compare|review|list|show)\b",
                text,
            )
        )
        and bool(
            re.search(
                r"\b(previous|historical|past|recurring)\b",
                text,
            )
        )
        and bool(
            re.search(
                r"\b(incidents?|failures?|problems?|"
                r"outages?|errors?|issues?)\b",
                text,
            )
        )
    )

    if (
        explicit_incident_search
        or historical_investigation
    ):
        return "search_incidents"

        # -----------------------------------------------------
    # Service health
    # -----------------------------------------------------

    service_status_intent = bool(
        re.search(
            r"\b(status|health|available|"
            r"availability|operational|"
            r"down|working|online|offline)\b",
            text,
        )
    )

    service_references = (
        find_service_identifiers(query)
    )

    instructional_request = bool(
        re.match(
            r"^(?:please\s+)?"
            r"(?:how|why|explain|define|"
            r"what does|what is meant by)\b",
            text,
        )
    )

    if (
        service_status_intent
        and service_references
        and not instructional_request
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
# 4A. NATURAL-LANGUAGE SERVICE RESOLUTION
# =========================================================


def normalize_service_text(
    text: str,
) -> str:
    """
    Normalize service names for alias matching.
    """

    text = text.casefold()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return " ".join(text.split())


def find_service_identifiers(
    query: str,
) -> set[str]:
    """
    Identify explicitly mentioned service IDs
    and natural-language service aliases.
    """

    explicit_ids = set(
        re.findall(
            SERVICE_ID_PATTERN,
            query.upper(),
        )
    )

    # Remove explicit identifiers before alias
    # matching to avoid accidental token matches.

    remaining_text = re.sub(
        SERVICE_ID_PATTERN,
        " ",
        query,
        flags=re.IGNORECASE,
    )

    normalized_query = (
        normalize_service_text(
            remaining_text
        )
    )

    matched_ids = set(explicit_ids)

    for service_id, aliases in (
        SERVICE_ALIASES.items()
    ):

        for alias in aliases:

            normalized_alias = (
                normalize_service_text(alias)
            )

            pattern = (
                rf"\b"
                rf"{re.escape(normalized_alias)}"
                rf"\b"
            )

            if re.search(
                pattern,
                normalized_query,
            ):

                matched_ids.add(service_id)
                break

    return matched_ids


def resolve_service_identifier(
    query: str,
) -> str | None:
    """
    Resolve exactly one known service identifier.

    Return None for unknown or ambiguous services.
    """

    identifiers = find_service_identifiers(
        query
    )

    if len(identifiers) != 1:
        return None

    service_id = next(iter(identifiers))

    if service_id not in SERVICE_ALIASES:
        return None

    return service_id

# =========================================================
# 5. OPERATIONAL SEARCH TERM EXTRACTION
# =========================================================


def extract_search_terms(
    query: str,
    route: RequestRoute,
) -> str:

    text = query.strip()

        # Normalize historical incident search requests.
    if route == "search_incidents":

        text = re.sub(
            r"^(?:please\s+)?"
            r"(?:investigate|analyze|compare|review)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\b(previous|historical|past|recurring)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\b(affecting|corporate)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        text = text.replace("-", " ")
        text = " ".join(text.split())

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
