#Implement Chroma vector storage
import shutil

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from incidentiq.core.config import settings
from incidentiq.rag.ingest import (
    load_knowledge_documents,
    chunk_documents,
    validate_chunks,
)


COLLECTION_NAME = "incidentiq_knowledge"


def get_embeddings():
    """Create the shared OpenAI embedding model."""

    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing from .env"
        )

    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )   #Create a shared OpenAI embedding model using the specified model and API key from the settings.


def get_vectorstore():
    """Open the persistent Chroma collection."""

    settings.incidentiq_vector_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(
            settings.incidentiq_vector_dir
        ),
    )  #Chroma stores the embeddings locally under .local/chroma.Unlike an in-memory database, the index remains available after the Python process exits.


def build_vectorstore(rebuild: bool = False):
    """
    Load documents, create chunks, and index them.

    rebuild=True replaces the local vector index.
    """

    documents = load_knowledge_documents()
    chunks = chunk_documents(documents)

    validate_chunks(documents, chunks)

    if rebuild:
        vector_dir = settings.incidentiq_vector_dir

        if vector_dir.exists():
            shutil.rmtree(vector_dir)

    vectorstore = get_vectorstore()

    chunk_ids = [
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected")

    # Chroma supports updating existing IDs.
    # Rebuild is preferred after source documents change.
    vectorstore.add_documents(
        documents=chunks,
        ids=chunk_ids,
    )   #We supply explicit IDs so that each stored chunk can be traced to its original document.We supply explicit IDs so that each stored chunk can be traced to its original document.

    print(f"Indexed documents: {len(documents)}")
    print(f"Indexed chunks: {len(chunks)}")

    return vectorstore


if __name__ == "__main__":
    build_vectorstore(rebuild=True)    #This deletes the configured local vector directory and rebuilds the index from our source documents.
