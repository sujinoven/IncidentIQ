#Ingestion Pipeline

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from incidentiq.core.config import settings


def load_knowledge_documents() -> list[Document]:
    """
    Load Markdown knowledge documents and attach metadata.
    """

    data_dir = settings.incidentiq_data_dir.resolve()

    metadata_path = data_dir / "metadata" / "documents.json"

    # Load the metadata records from the JSON file.
    metadata_records = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    documents = []

    for record in metadata_records:
        content_path = (
            data_dir / record["content_file"]
        ).resolve()

        # Prevent metadata paths from escaping the data directory.
        if not content_path.is_relative_to(data_dir):
            raise ValueError(
                f"Unsafe document path: {content_path}"
            )

        content = content_path.read_text(
            encoding="utf-8"
        )

        if not content.strip():
            raise ValueError(
                f"Empty document: {record['document_id']}"
            )
        #Create a langchain Document object with the content and metadata.
        document = Document(
            page_content=content,
            metadata={
                "document_id": record["document_id"],
                "title": record["title"],
                "document_type": record["document_type"],
                "service_id": record["service_id"],
                "classification": record["classification"],
                "version": record["version"],
                "status": record["status"],
                "owner_team": record["owner_team"],
                "source": str(content_path),
            },
        )

        documents.append(document)

    return documents


def chunk_documents(
    documents: list[Document],
) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=[
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )

    chunks = []

    for document in documents:
        document_chunks = splitter.split_documents([document])

        for index, chunk in enumerate(document_chunks):
            document_id = document.metadata["document_id"]

            chunk.metadata["chunk_id"] = (
                f"{document_id}-CHUNK-{index:04d}"
            )

            chunks.append(chunk)

    return chunks


#Add a security consistency check.We should verify that chunking never loses the classification metadata.
def validate_chunks(
    documents: list[Document],
    chunks: list[Document],
) -> None:
    """
    Verify every chunk retains its source classification.
    """

    classifications = {
        doc.metadata["document_id"]: doc.metadata["classification"]
        for doc in documents
    }

    for chunk in chunks:
        document_id = chunk.metadata["document_id"]

        if document_id not in classifications:
            raise ValueError(
                f"Unknown chunk document: {document_id}"
            )

        expected = classifications[document_id]
        actual = chunk.metadata.get("classification")

        if actual != expected:
            raise ValueError(
                f"Classification mismatch: {document_id}"
            )

    print("PASS: All chunks retain their classifications")



def main():
    documents = load_knowledge_documents()
    chunks = chunk_documents(documents)
    #This check ensures every chunk retains the classification assigned to its original document.
    validate_chunks(documents, chunks)

    print(f"Loaded documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")

    if chunks:
        first = chunks[0]

        print("\nSample chunk metadata:")
        print(
            json.dumps(
                first.metadata,
                indent=2,
            )
        )

        print("\nSample chunk content:")
        print(first.page_content[:400])


if __name__ == "__main__":
    main()
