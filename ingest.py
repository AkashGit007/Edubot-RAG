"""
EduBot Incremental Knowledge Ingestion

Supports:
- TXT
- Markdown
- PDF
- HTML
- HTM

Features:
- Incremental ingestion
- SHA256 document hashing
- Duplicate detection
- Changed-document replacement
- Qdrant vector storage
- Document metadata
"""

from pathlib import Path
import hashlib
import json
import re
import uuid
from fastembed import SparseTextEmbedding
from openai import files
from qdrant_client import QdrantClient, models

from bs4 import BeautifulSoup
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

from config import (
    DATA_DIR,
    QDRANT_URL,
    QDRANT_COLLECTION,
    CHUNK_WORDS,
    CHUNK_OVERLAP,
)

from embeddings import embed


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".html",
    ".htm",
}

MANIFEST_PATH = DATA_DIR / ".ingestion_manifest.json"

BATCH_SIZE = 100
SPARSE_MODEL_NAME = "Qdrant/bm25"

sparse_model = SparseTextEmbedding(
    model_name=SPARSE_MODEL_NAME
)

# ============================================================
# QDRANT
# ============================================================

client = QdrantClient(
    url=QDRANT_URL,
)


# ============================================================
# TEXT EXTRACTION
# ============================================================

def clean_text(text: str) -> str:

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def extract_txt(path: Path) -> str:

    return path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


def extract_pdf(path: Path) -> str:

    reader = PdfReader(
        str(path),
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        try:

            text = page.extract_text() or ""

            if text.strip():

                pages.append(
                    f"[Page {page_number}]\n{text}"
                )

        except Exception as exc:

            print(
                f"Warning: failed to read "
                f"{path.name} page {page_number}: "
                f"{exc}"
            )

    return "\n\n".join(pages)


def extract_html(path: Path) -> str:

    html = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    for element in soup(
        [
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
        ]
    ):
        element.decompose()

    return soup.get_text(
        separator="\n",
    )


def extract_text(path: Path) -> str:

    extension = path.suffix.lower()

    if extension in {
        ".txt",
        ".md",
    }:
        return extract_txt(path)

    if extension == ".pdf":
        return extract_pdf(path)

    if extension in {
        ".html",
        ".htm",
    }:
        return extract_html(path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text: str,
    chunk_words: int = CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP,
):

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_words,
            len(words),
        )

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


# ============================================================
# HASHING
# ============================================================

def calculate_file_hash(path: Path) -> str:

    sha256 = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            data = file.read(
                1024 * 1024
            )

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# DOCUMENT ID
# ============================================================

def create_document_id(
    relative_path: str,
) -> str:

    digest = hashlib.sha256(
        relative_path.encode("utf-8")
    ).hexdigest()

    return digest[:32]


# ============================================================
# POINT ID
# ============================================================

def create_point_id(
    document_id: str,
    chunk_id: int,
) -> str:

    value = (
        f"{document_id}:"
        f"{chunk_id}"
    )

    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()

    return str(
        uuid.UUID(
            digest[:32]
        )
    )


# ============================================================
# MANIFEST
# ============================================================

def load_manifest():

    if not MANIFEST_PATH.exists():

        return {}

    try:

        data = json.loads(
            MANIFEST_PATH.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, dict):

            return data

        return {}

    except Exception:

        print(
            "Warning: manifest could not be read."
        )

        return {}


def save_manifest(manifest):

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

     
# ============================================================
# QDRANT COLLECTION
# ============================================================

def collection_exists():

    collections = (
        client
        .get_collections()
        .collections
    )

    return any(
        collection.name
        == QDRANT_COLLECTION
        for collection in collections
    )


def create_collection_if_needed(
    vector_size: int,
):

    if collection_exists():

        return

    print(
        f"Creating Qdrant collection: "
        f"{QDRANT_COLLECTION}"
    )

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


# ============================================================
# DELETE DOCUMENT
# ============================================================

def delete_document(
    document_id: str,
):

    if not collection_exists():

        return

    print(
        f"Removing old chunks for "
        f"document: {document_id}"
    )

    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(
                            value=document_id
                        ),
                    )
                ]
            )
        ),
        wait=True,
    )


# ============================================================
# UPSERT DOCUMENT
# ============================================================

def upsert_document(
    records,
    vectors,
):

    points = []

    for record, vector in zip(
        records,
        vectors,
    ):

        point_id = create_point_id(
            record["document_id"],
            record["chunk_id"],
        )

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload={
                    "document_id":
                        record["document_id"],

                    "document_hash":
                        record["document_hash"],

                    "source":
                        record["source"],

                    "path":
                        record["path"],

                    "file_type":
                        record["file_type"],

                    "chunk_id":
                        record["chunk_id"],

                    "total_chunks":
                        record["total_chunks"],

                    "text":
                        record["text"],
                },
            )
        )

    for start in range(
        0,
        len(points),
        BATCH_SIZE,
    ):

        batch = points[
            start:start + BATCH_SIZE
        ]

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=batch,
            wait=True,
        )

        print(
            f"Uploaded "
            f"{min(start + BATCH_SIZE, len(points))}"
            f"/{len(points)} chunks"
        )


# ============================================================
# PROCESS ONE DOCUMENT
# ============================================================

def process_document(
    path: Path,
    manifest,
):

    relative_path = str(
        path.relative_to(DATA_DIR)
    )

    document_id = create_document_id(
        relative_path
    )

    file_hash = calculate_file_hash(
        path
    )

    previous = manifest.get(
        relative_path
    )

    # --------------------------------------------------------
    # UNCHANGED DOCUMENT
    # --------------------------------------------------------

    if (
        previous
        and previous.get("hash")
        == file_hash
    ):

        print(
            f"\nSKIP: {relative_path}"
        )

        print(
            "Reason: document unchanged."
        )

        return False

    # --------------------------------------------------------
    # NEW OR CHANGED DOCUMENT
    # --------------------------------------------------------

    if previous:

        print(
            f"\nUPDATE: {relative_path}"
        )

        delete_document(
            document_id
        )

    else:

        print(
            f"\nNEW: {relative_path}"
        )

    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------

    raw_text = extract_text(
        path
    )

    text = clean_text(
        raw_text
    )

    if not text:

        print(
            "Skipping empty document."
        )

        return False

    # --------------------------------------------------------
    # CHUNK
    # --------------------------------------------------------

    chunks = chunk_text(
        text
    )

    if not chunks:

        print(
            "Skipping document with "
            "no usable chunks."
        )

        return False

    print(
        f"Chunks created: {len(chunks)}"
    )

    # --------------------------------------------------------
    # RECORDS
    # --------------------------------------------------------

    records = []

    total_chunks = len(chunks)

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        records.append(
            {
                "document_id":
                    document_id,

                "document_hash":
                    file_hash,

                "source":
                    path.name,

                "path":
                    relative_path,

                "file_type":
                    path.suffix.lower(),

                "chunk_id":
                    index,

                "total_chunks":
                    total_chunks,

                "text":
                    chunk,
            }
        )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    print(
        "Generating embeddings..."
    )

    texts = [
        record["text"]
        for record in records
    ]

    vectors = embed(
        texts
    )

    # --------------------------------------------------------
    # CREATE COLLECTION
    # --------------------------------------------------------

    create_collection_if_needed(
        vectors.shape[1]
    )

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    upsert_document(
        records,
        vectors,
    )

    # --------------------------------------------------------
    # UPDATE MANIFEST
    # --------------------------------------------------------

    manifest[
        relative_path
    ] = {
        "hash": file_hash,
        "document_id": document_id,
        "chunks": total_chunks,
        "file_type":
            path.suffix.lower(),
    }

    return True


# ============================================================
# REMOVE DELETED DOCUMENTS
# ============================================================

def remove_deleted_documents(
    manifest,
):

    current_files = {
        str(
            path.relative_to(DATA_DIR)
        )
        for path in DATA_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    }

    deleted = (
        set(manifest.keys())
        - current_files
    )

    for relative_path in deleted:

        entry = manifest[
            relative_path
        ]

        document_id = entry.get(
            "document_id"
        )

        if document_id:

            print(
                f"\nREMOVED: {relative_path}"
            )

            delete_document(
                document_id
            )

        del manifest[
            relative_path
        ]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "EduBot Incremental Knowledge Ingestion"
    )
    print("=" * 60)

    print(
        f"\nKnowledge directory:"
        f"\n{DATA_DIR}"
    )

    print(
        f"\nQdrant:"
        f"\n{QDRANT_URL}"
    )

    print(
        f"\nCollection:"
        f"\n{QDRANT_COLLECTION}"
    )

    # --------------------------------------------------------
    # Ensure directory exists
    # --------------------------------------------------------

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find documents
    # --------------------------------------------------------

    files = sorted(
        path
        for path in DATA_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )

    if not files:

        print(
            "\nNo supported documents found."
        )

        print(
            "Supported:"
            " TXT, MD, PDF, HTML, HTM"
        )

        return

    print(
        f"\nFound {len(files)} document(s)."
    )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    manifest = load_manifest() or {}

# --------------------------------------------------------
# Rebuild if Qdrant collection is missing
# --------------------------------------------------------

if not collection_exists():

    print(
        "\nQdrant collection does not exist."
    )

    print(
        "Rebuilding knowledge base from documents..."
    )

    manifest = {}
    # --------------------------------------------------------
    # Process documents
    # --------------------------------------------------------

    processed = 0
    skipped = 0

    for path in files:

        try:

            changed = process_document(
                path,
                manifest,
            )

            if changed:

                processed += 1

            else:

                skipped += 1

        except Exception as exc:

            print(
                f"\nERROR: {path}"
            )

            print(
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # Remove deleted files
    # --------------------------------------------------------

    remove_deleted_documents(
        manifest
    )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    save_manifest(
        manifest
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    print(
        f"\nProcessed: {processed}"
    )

    print(
        f"Skipped unchanged: {skipped}"
    )

    print(
        f"Tracked documents: "
        f"{len(manifest)}"
    )

    print(
        f"\nQdrant collection:"
        f" {QDRANT_COLLECTION}"
    )

    print(
        "\nDashboard:"
        "\nhttp://localhost:6333/dashboard"
    )


if __name__ == "__main__":
    main()