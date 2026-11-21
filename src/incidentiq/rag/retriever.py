#building a classification-aware retrieval

from langchain_core.documents import Document

from incidentiq.core.models import UserContext
from incidentiq.core.security import ROLE_PERMISSIONS
from incidentiq.rag.vectorstore import get_vectorstore


def retrieve_knowledge(
    query: str,
    user: UserContext,
    k: int = 4,
) -> list[Document]:
    """
    Retrieve semantically relevant documents that
    the authenticated user's role may access.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if not 1 <= k <= 20:
        raise ValueError("k must be between 1 and 20")

    allowed_classifications = sorted(
        ROLE_PERMISSIONS[user.role]
    )

    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search(
        query=query,
        k=k,
        filter={
            "classification": {
                "$in": allowed_classifications
            }
        },
    )

    # Additional defense: verify returned metadata.
    for document in results:
        classification = document.metadata.get(
            "classification"
        )

        if classification not in allowed_classifications:
            raise PermissionError(
                "Vector retrieval returned unauthorized content"
            )

    return results
