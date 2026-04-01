import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from backend.config.settings import QDRANT_URL, QDRANT_COLLECTION, GEMINI_EMBEDDING_MODEL
import uuid

log = structlog.get_logger()

_qdrant_client = None
_embedding_model = None

def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL)
        log.info("qdrant.connected", url=QDRANT_URL)
    return _qdrant_client

def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)
    return _embedding_model

def ensure_collection():
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        log.info("qdrant.collection.created", collection=QDRANT_COLLECTION)

def store_embeddings_qdrant(video_id: str, chunks: list) -> bool:
    """Embed chunks and upsert into Qdrant under the given video_id."""
    try:
        ensure_collection()
        client = get_qdrant_client()
        model = get_embedding_model()

        texts = [c["Text"] for c in chunks]
        log.info("qdrant.embedding.start", video_id=video_id, chunks=len(texts))
        vectors = model.embed_documents(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "video_id": video_id,
                    "text": texts[i],
                    "start": chunks[i].get("Start", ""),
                    "end": chunks[i].get("End", "")
                }
            )
            for i, vec in enumerate(vectors)
        ]

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        log.info("qdrant.embedding.done", video_id=video_id, points=len(points))
        return True
    except Exception as e:
        log.error("qdrant.embedding.failed", video_id=video_id, error=str(e))
        return False

def search_qdrant(video_id: str, query: str, top_k: int = 5) -> list:
    """Search Qdrant for top_k relevant chunks for a given video_id and query."""
    try:
        client = get_qdrant_client()
        model = get_embedding_model()

        query_vector = model.embed_query(query)

        results = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            ),
            limit=top_k,
            with_payload=True
        )

        return [
            {
                "text": r.payload["text"],
                "start": r.payload.get("start", ""),
                "end": r.payload.get("end", ""),
                "score": r.score
            }
            for r in results
        ]
    except Exception as e:
        log.error("qdrant.search.failed", video_id=video_id, error=str(e))
        return []

def is_video_indexed(video_id: str) -> bool:
    """Check if a video already has vectors in Qdrant."""
    try:
        client = get_qdrant_client()
        results = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            ),
            limit=1
        )
        return len(results[0]) > 0
    except Exception:
        return False

def delete_video_vectors(video_id: str):
    """Delete all vectors for a given video from Qdrant."""
    try:
        client = get_qdrant_client()
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            )
        )
        log.info("qdrant.vectors.deleted", video_id=video_id)
    except Exception as e:
        log.error("qdrant.delete.failed", video_id=video_id, error=str(e))

def ping_qdrant() -> bool:
    try:
        client = get_qdrant_client()
        client.get_collections()
        return True
    except Exception:
        return False