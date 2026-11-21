#Implement role-based access control

from incidentiq.core.models import (
    Classification,
    UserContext,
    UserRole,
)


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
