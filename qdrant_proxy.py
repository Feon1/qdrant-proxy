"""Read-only прокси к Qdrant. Только поиск, никакой записи."""
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from openai import OpenAI

# ===== Настройки из env =====
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "xiaozhi_knowledge_max")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "5"))
MAX_LIMIT = int(os.getenv("MAX_LIMIT", "20"))

# Простой токен для клиентов (чтобы прокси не был совсем открытым)
PROXY_TOKEN = os.getenv("PROXY_TOKEN", "")  # если пусто — авторизация отключена

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Qdrant Read-Only Proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ===== Модель запроса =====
class SearchRequest(BaseModel):
    query: str
    limit: int = DEFAULT_LIMIT


# ===== Проверка токена клиента =====
def check_token(request: Request):
    if not PROXY_TOKEN:
        return  # токен не задан — прокси открыт
    token = request.headers.get("x-proxy-token", "")
    if token != PROXY_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid proxy token")


# ===== Embeddings через OpenAI =====
def embed(text: str) -> list[float]:
    resp = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=[text],
        dimensions=EMBED_DIM,  # для text-embedding-3-* можно указать
    )
    return resp.data[0].embedding


# ===== Основной эндпоинт =====
@app.post("/search")
async def search(req: SearchRequest, request: Request):
    check_token(request)

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    limit = min(max(1, req.limit), MAX_LIMIT)

    # 1. Эмбеддинг запроса
    try:
        vector = embed(req.query)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding error: {e}")

    # 2. Поиск в Qdrant
    try:
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=limit,
            with_payload=True,
        )
    except Exception as e:
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


# ===== Health =====
@app.get("/health")
async def health():
    try:
        cols = qdrant.get_collections()
        return {
            "status": "ok",
            "collections": [c.name for c in cols.collections],
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ===== Список коллекций (только имена) =====
@app.get("/collections")
async def collections(request: Request):
    check_token(request)
    try:
        cols = qdrant.get_collections()
        return {"collections": [c.name for c in cols.collections]}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8200))
    uvicorn.run(app, host="0.0.0.0", port=port)
