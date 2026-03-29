from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os

from src.routers import public_router

# Создание приложения FastAPI
app = FastAPI(
    title="Public RAG API",
    description="Внешний API для RAG системы",
    version="1.0.0",
    docs_url="/docs"
)

# CORS middleware (настройте под свои нужды)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутеров
app.include_router(public_router.router)


# Базовые эндпоинты
@app.get("/", tags=["Root"])
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Public RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }

# Обработчик глобальных исключений
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )
