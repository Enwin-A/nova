"""ChromaDB vector store and shared embedding model.

chromadb is imported eagerly to avoid Windows DLL load-order segfaults
when it gets imported after pandas/onnxruntime in the same process.
"""

from __future__ import annotations

from typing import Any

import chromadb  # Must stay at top level — see module docstring

import config

COLLECTION_NAME = "reviews"
_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    return _embedding_model


def get_chroma_client(path: str | None = None) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=path or config.CHROMA_PATH)


def get_collection(client: chromadb.PersistentClient | None = None):
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_passages(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    model = get_embedding_model()
    prefixed = [f"passage: {text}" for text in texts]
    embeddings = model.encode(
        prefixed,
        batch_size=batch_size or config.EMBED_BATCH_SIZE,
        show_progress_bar=len(texts) > 32,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    model = get_embedding_model()
    embedding = model.encode(
        f"query: {query}",
        normalize_embeddings=True,
    )
    return embedding.tolist()


def build_where_clause(
    language_filter: str | None = None,
    rating_filter: int | None = None,
    country_filter: str | None = None,
    app_type_filter: str | None = None,
    sarcasm_filter: bool | None = None,
) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if language_filter:
        clauses.append({"language": {"$eq": str(language_filter)}})
    if rating_filter is not None:
        clauses.append({"rating": {"$eq": int(rating_filter)}})
    if country_filter:
        clauses.append({"country": {"$eq": str(country_filter)}})
    if app_type_filter:
        clauses.append({"app_type": {"$eq": str(app_type_filter)}})
    if sarcasm_filter is not None:
        clauses.append({"sarcasm": {"$eq": 1 if sarcasm_filter else 0}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def upsert_reviews(
    review_ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    embeddings: list[list[float]] | None = None,
) -> None:
    collection = get_collection()
    if embeddings is None:
        embeddings = embed_passages(documents)
    collection.upsert(
        ids=review_ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def update_review_metadata(review_ids: list[str], metadatas: list[dict[str, Any]]) -> None:
    collection = get_collection()
    collection.update(ids=review_ids, metadatas=metadatas)


def query_reviews(
    query: str,
    max_results: int | None = None,
    language_filter: str | None = None,
    rating_filter: int | None = None,
    country_filter: str | None = None,
    app_type_filter: str | None = None,
    sarcasm_filter: bool | None = None,
) -> list[dict[str, Any]]:
    collection = get_collection()
    where = build_where_clause(
        language_filter,
        rating_filter,
        country_filter,
        app_type_filter,
        sarcasm_filter,
    )
    query_embedding = embed_query(query)

    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": max_results or config.TOP_K_RESULTS,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    parsed: list[dict[str, Any]] = []
    if not results["ids"] or not results["ids"][0]:
        return parsed

    for idx, review_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][idx]
        score = 1.0 - distance
        metadata = results["metadatas"][0][idx] or {}
        document = results["documents"][0][idx] or ""
        parsed.append(
            {
                "review_id": review_id,
                "score": round(score, 4),
                "original_text": document,
                "rating": metadata.get("rating"),
                "language": metadata.get("language"),
                "country": metadata.get("country"),
                "date": metadata.get("date"),
                "app_type": metadata.get("app_type"),
                "topic": metadata.get("topic"),
                "sentiment": metadata.get("sentiment"),
                "sarcasm": metadata.get("sarcasm"),
                "sarcasm_confidence": metadata.get("sarcasm_confidence"),
            }
        )

    return parsed
