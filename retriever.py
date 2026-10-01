from qdrant_client import QdrantClient, models

from config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    TOP_K,
    MIN_SCORE,
)

from embeddings import embed


class Retriever:

    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        self.collection_name = QDRANT_COLLECTION

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        min_score: float = MIN_SCORE,
        file_type: str | None = None,
        source: str | None = None,
        document_id: str | None = None,
    ):
        """
        Semantic search with optional metadata filtering.

        Examples:

        search("What is photosynthesis?")

        search(
            "Explain artificial intelligence",
            file_type="txt"
        )

        search(
            "Explain Python",
            source="test_python.md"
        )
        """

        if not query or not query.strip():
            return []

        query_vector = embed([query])[0]

        # Build optional Qdrant filters
        conditions = []

        if file_type:
            conditions.append(
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value=file_type),
                )
            )

        if source:
            conditions.append(
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=source),
                )
            )

        if document_id:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            )

        query_filter = None

        if conditions:
            query_filter = models.Filter(
                must=conditions
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            query_filter=query_filter,
            limit=top_k,
            score_threshold=min_score,
            with_payload=True,
        ).points

        passages = []

        for result in results:

            payload = result.payload or {}

            passages.append(
                {
                    "text": payload.get("text", ""),
                    "source": payload.get("source", "Unknown"),
                    "path": payload.get("path", ""),
                    "file_type": payload.get("file_type", ""),
                    "document_id": payload.get("document_id", ""),
                    "chunk_id": payload.get("chunk_id", 0),
                    "total_chunks": payload.get("total_chunks", 0),
                    "score": float(result.score),
                }
            )

        return passages


if __name__ == "__main__":

    retriever = Retriever()

    question = input("\nAsk EduBot something: ")

    results = retriever.search(question)

    if not results:
        print("\nNo relevant information found.")

    else:
        print("\nSearch Results")
        print("=" * 60)

        for i, result in enumerate(results, start=1):

            print(f"\nResult {i}")
            print(f"Source: {result['source']}")
            print(f"File type: {result['file_type']}")
            print(f"Score: {result['score']:.4f}")

            print("\nText:")
            print(result["text"][:500])

            print("-" * 60)