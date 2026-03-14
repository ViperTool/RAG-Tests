from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any
from contextlib import asynccontextmanager

from src.services.rag import RAGEngine
import config

rag_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    print("Инициализация RAG-моделей (GPU/CPU)...")
    rag_engine = RAGEngine(collection_name=config.COLLECTION_NAME)
    yield
    print("Выгрузка моделей...")


app = FastAPI(title="Internal RAG Core", lifespan=lifespan)


class InternalQuery(BaseModel):
    query: str
    chunking_config: Optional[dict[str, Any]] = None


@app.get("/internal/generate")
async def generate_answer(request: InternalQuery):
    """Единственный эндпоинт для связи с публичным API"""
    try:
        chunking_cfg = request.chunking_config or config.SELECTED_C_CONFIG
        context = rag_engine.search(request.query, chunkung_config=chunking_cfg)

        if config.G_USE_REMOTE_MODEL:
            result = rag_engine.generator_service.ask_api(request.query, context)
        else:
            result = rag_engine.generator_service.generate_response(request.query, context)

        # Обработка кортежа, если возвращаются токены
        if isinstance(result, tuple):
            answer = result[0]
        else:
            answer = str(result)

        return {
            "answer": answer,
            "context": context,
            "chunks_count": len(context.split('\n')) if context else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
