"""Read-only прокси к Qdrant. Только поиск, никакой записи."""
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
import httpx


# ============================================================
# НАСТРОЙКИ
# ============================================================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "xiaozhi_knowledge_max")

DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "5"))
MAX_LIMIT = int(os.getenv("MAX_LIMIT", "20"))
PROXY_TOKEN = os.getenv("PROXY_TOKEN", "")

# Провайдер эмбеддингов: "polza" (OpenAI-совместимый) или "jina"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "polza").lower()

# Polza AI (OpenAI-совместимый эндпоинт)
POLZA_API_KEY = os.getenv("POLZA_EMBEDDING_API_KEY") or os.getenv("POLZA_API_KEY")
POLZA_BASE_URL = os.getenv("POLZA_BASE_URL", "https://api.polza.ai/api/v1")
POLZA_MODEL = os.getenv("POLZA_EMBED_MODEL", "text-embedding-3-small")

# Jina AI (если понадобится переключиться)
JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_API_URL = os.getenv("JINA_API_URL", "https://api.jina.ai/v1/embeddings")
JINA_MODEL = os.getenv("JINA_MODEL", "jina-embeddings-v3")

# Размерность вектора — ДОЛЖНА СОВПАДАТЬ с коллекцией Qdrant!
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))


# ============================================================
# КЛИЕНТЫ
# ============================================================
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

app = FastAPI(title="Qdrant Read-Only Proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    limit: int = DEFAULT_LIMIT


def check_token(request: Request):
    if not PROXY_TOKEN:
        return
    token = request.headers.get("x-proxy-token", "")
    if token != PROXY_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid proxy token")


# ============================================================
# EMBEDDINGS
# ============================================================
async def embed_polza(text: str) -> list[float]:
    """Эмбеддинг через Polza AI (OpenAI-совместимый /embeddings)."""
    if not POLZA_API_KEY:
        raise RuntimeError("POLZA_EMBEDDING_API_KEY не задан")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {POLZA_API_KEY}",
    }
    payload = {
        "model": POLZA_MODEL,
        "input": [text],
    }
    # dimensions передаём только если нужно явно указать (не 1536 по умолчанию)
    if EMBED_DIM and EMBED_DIM != 1536:
        payload["dimensions"] = EMBED_DIM

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{POLZA_BASE_URL}/embeddings",
            headers=headers,
            json=payload,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Polza API error {r.status_code}: {r.text[:200]}")
        data = r.json()
        return data["data"][0]["embedding"]


async def embed_jina(text: str) -> list[float]:
    """Эмбеддинг через Jina AI."""
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY не задан")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
    }
    payload = {
        "model": JINA_MODEL,
        "input": [text],
        "task": "text-matching",
        "dimensions": EMBED_DIM,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(JINA_API_URL, headers=headers, json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"Jina API error {r.status_code}: {r.text[:200]}")
        data = r.json()
        return data["data"][0]["embedding"]


async def embed(text: str) -> list[float]:
    """Роутер провайдеров."""
    if EMBEDDING_PROVIDER == "jina":
        return await embed_jina(text)
    return await embed_polza(text)


# ============================================================
# ЭНДПОИНТЫ
# ============================================================
@app.post("/search")
async def search(req: SearchRequest, request: Request):
    check_token(request)

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    limit = min(max(1, req.limit), MAX_LIMIT)

    # 1. Эмбеддинг
    try:
        vector = await embed(req.query)
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        raise HTTPException(status_code=502, detail=f"Embedding error: {e}")

    print(f"🔍 [{request.client.host}] query='{req.query[:60]}' limit={limit} dim={len(vector)}")

    # Проверка размерности — критично!
    if len(vector) != EMBED_DIM:
        print(f"⚠️ Размерность не совпадает: получено {len(vector)}, ожидалось {EMBED_DIM}")

    # 2. Поиск в Qdrant
    try:
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=limit,
            with_payload=True,
        )
    except Exception as e:
        print(f"❌ Qdrant error: {e}")
        raise HTTPException(status_code=502, detail=f"Qdrant error: {e}")

    # 3. Формируем ответ
    chunks = []
    for hit in results:
        payload = hit.payload or {}
        chunks.append({
            "score": float(hit.score),
            "text": payload.get("text", ""),
            "source": payload.get("source", ""),
        })

    return {
        "query": req.query,
        "collection": COLLECTION_NAME,
        "count": len(chunks),
        "chunks": chunks,
    }


@app.get("/health")
async def health():
    try:
        cols = qdrant.get_collections()
        return {
            "status": "ok",
            "provider": EMBEDDING_PROVIDER,
            "dim": EMBED_DIM,
            "collection": COLLECTION_NAME,
            "collections_count": len(cols.collections),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/collections")
async def collections(request: Request):
    """Показать все коллекции с их размерностями и количеством точек."""
    check_token(request)
    try:
        cols = qdrant.get_collections()
        out = []
        for c in cols.collections:
            info = qdrant.get_collection(c.name)
            out.append({
                "name": c.name,
                "points": info.points_count,
                "dim": info.config.params.vectors.size,
            })
        return {"collections": out}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8200))
    print("=" * 60)
    print(f"  Qdrant Read-Only Proxy (port {port})")
    print(f"  Provider: {EMBEDDING_PROVIDER}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Dim: {EMBED_DIM}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=port)
