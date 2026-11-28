#Implement role-based access control

from incidentiq.core.models import (
    Classification,
    UserContext,
    UserRole,
)
import re


# Explicit permissions for each role.
ROLE_PERMISSIONS: dict[
    UserRole, set[Classification]
] = {
    "employee": {
        "employee_public",
    },
    "it_support": {
        "employee_public",
        "it_internal",
    },
    "it_admin": {
        "employee_public",
        "it_internal",
        "admin_restricted",
    },
}


class AccessDeniedError(PermissionError):
    """Raised when a user lacks permission."""


def can_access(
    role: UserRole,
    classification: Classification,
) -> bool:
    """Check whether a role may access a classification."""

    return classification in ROLE_PERMISSIONS[role]


def require_access(
    user: UserContext,
    classification: Classification,
) -> None:
    """Raise an error if access is not permitted."""

    if not can_access(user.role, classification):
        raise AccessDeniedError(
            "You do not have permission to access "
            "this resource."
        )


def filter_authorized_documents(
    user: UserContext,
    documents: list[dict],
) -> list[dict]:
    """Return only documents authorized for the user."""

    authorized = []

    for document in documents:
        classification = document.get("classification")

        # Missing or unknown classifications are denied.
        if classification not in ROLE_PERMISSIONS[user.role]:
            continue

        authorized.append(document)

    return authorized


#Centralizing input guardrails for security checks in one place to avoid duplication and ensure consistency across the application.


# =========================================================
# SHARED INPUT SECURITY GUARDRAILS
# =========================================================




RESTRICTED_ACTION_PATTERNS = (

    # Destructive operations.

    r"^(?:please\s+)?(?:delete|remove|purge)\s+"
    r"(?:(?:all|the|a|an)\s+)?"
    r"(?:tickets?|incidents?|assets?|records?|logs?|databases?)\b",
    # Block direct infrastructure manipulation requests.
    r"^(?:please\s+)?(?:restart|shutdown|shut\s+down|reboot)\s+(?:(?:the|all|our|a|an|production|prod|staging|development|dev|critical|database|application|web)\s+)*(?:servers?|services?|systems?|instances?|hosts?|svc-[a-z0-9-]+)\b",

    # Infrastructure changes.

    r"^(?:please\s+)?(?:restart|shutdown|shut down|reboot)\s+"
    r"(?:(?:the|all|a)\s+)?"
    r"(?:servers?|services?|production systems?|svc-)",

    # Security control bypass.

    r"^(?:please\s+)?(?:disable|bypass|turn off)\s+"
    r"(?:(?:the|all)\s+)?"
    r"(?:mfa|authentication|rbac|access controls?|security controls?)\b",

    # Privilege escalation.

    r"^(?:please\s+)?(?:grant|assign|elevate)\s+"
    r".*\b(?:admin|administrator|root)\b",

    # Sensitive information disclosure.

    r"^(?:please\s+)?(?:reveal|show|export|dump)\s+"
    r"(?:(?:all|the|your|system)\s+)*"
    r"(?:passwords?|api keys?|credentials?|secrets?|system prompts?)\b",

    # Unsupported ticket and incident modifications.

    r"^(?:please\s+)?(?:close|resolve|update|modify)\s+"
    r"(?:(?:the|a|an)\s+)?"
    r"(?:tickets?|incidents?|tkt-|inc-)",

    # Confirmation must occur through the approved UI.

    r"^(?:please\s+)?(?:confirm|execute)\s+"
    r".*\bticket\b",

    # Reject explicit attempts to bypass ticket approval.
    r"\b(?:do\s+not|don't|never)\s+(?:ask\s+for|request|require|wait\s+for)\s+(?:human\s+)?(?:approval|confirmation)\b.*\b(?:create|submit|execute|confirm)\b.*\btickets?\b",

    # Reject direct ticket execution without approval.
    r"^(?:please\s+)?(?:create|submit|execute|confirm)\b.*\btickets?\b.*\b(?:without|bypassing|skipping)\s+(?:explicit\s+)?(?:human\s+)?(?:approval|confirmation)\b",

    # Reject explicit invocation of the protected creation tool.
    r"^(?:please\s+)?(?:invoke|call|execute|run|use)\s+(?:the\s+)?create_ticket\b",
)



def check_input_guardrails(
    query: str,
) -> tuple[str, str] | None:

    """
    Validate an enterprise IT request.

    Returns:
        None when the request passes validation.

        A tuple containing:
            error_code,
            user_facing_message

        when the request is rejected.
    """

    if not isinstance(query, str):

        return (
            "INVALID_INPUT",
            "Please provide a valid IT support request.",
        )

    query = query.strip()

    if len(query) < 3 or len(query) > 2000:

        return (
            "INVALID_INPUT",
            "The request must contain 3 to 2000 characters.",
        )

    # Reject unexpected control characters.

    if any(
        ord(character) < 32
        and character not in "\n\r\t"
        for character in query
    ):

        return (
            "INVALID_INPUT",
            "The request contains invalid characters.",
        )

    normalized_query = " ".join(
        query.lower().split()
    )

    # Detect explicitly unsupported actions.

    for pattern in RESTRICTED_ACTION_PATTERNS:

        if re.search(
            pattern,
            normalized_query,
            flags=re.IGNORECASE,
        ):

            return (
                "RESTRICTED_ACTION",

                (
                    "This operation is not supported by "
                    "IncidentIQ. Please contact an authorized "
                    "IT administrator or use the approved "
                    "enterprise workflow."
                ),
            )

    return None
