from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import sys

import config
from src.routers import rag_router
from src.services.rag import RAGEngine
from src.utils.logger import init_logging
from src.schemas.response_models import HealthResponse

# Инициализация логирования
init_logging()
logger = logging.getLogger(__name__)


# Lifespan для управления жизненным циклом приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление ресурсами при запуске и остановке"""
    # Startup
    logger.info("Запуск приложения...")
    try:
        rag_router.rag_engine = RAGEngine(collection_name=config.COLLECTION_NAME)
        logger.info("RAG-система успешно инициализирована")
    except Exception as e:
        logger.error(f"Ошибка инициализации RAG: {e}")
        sys.exit(1)

    yield

    # Shutdown
    logger.info("Остановка приложения...")
    # Здесь можно добавить очистку ресурсов
    logger.info("Приложение остановлено")


# Создание приложения FastAPI
app = FastAPI(
    title="Outer Wilds RAG API",
    description="API для интеллектуального ассистента по базе знаний игры Outer Wilds",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
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
app.include_router(rag_router.router)


# Базовые эндпоинты
@app.get("/", tags=["Root"])
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Outer Wilds RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Проверка состояния сервиса"
)
async def health_check():
    """Проверка работоспособности API и загруженных моделей"""
    try:
        rag = rag_router.rag_engine

        return HealthResponse(
            status="healthy",
            models_loaded={
                "embedding_model": rag.embedding_service.model is not None,
                "generator_model": rag.generator_service.model is not None if not config.G_USE_REMOTE_MODEL else "remote",
                "ner_model": True,
                "crossencoder_model": True
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


# Обработчик глобальных исключений
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Необработанное исключение: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
