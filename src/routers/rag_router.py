from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
import logging
from typing import Optional

from src.schemas.request_models import QueryRequest
from src.schemas.response_models import QueryResponse, ErrorResponse
from src.services.rag import RAGEngine
import config

logger = logging.getLogger(__name__)

# Создание роутера
router = APIRouter(
    prefix="/api/v1/rag",
    tags=["RAG"],
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        400: {"model": ErrorResponse, "description": "Bad Request"}
    }
)

# Глобальный экземпляр RAG (инициализируется при старте)
rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Dependency для получения RAG engine"""
    if rag_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG система не инициализирована"
        )
    return rag_engine


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Задать вопрос RAG-системе",
    description="Отправляет запрос в RAG-систему и получает ответ на основе базы знаний Outer Wilds"
)
async def query_rag(request: QueryRequest):
    """
    Обработка запроса к RAG-системе

    - **query**: Вопрос пользователя (обязательно)
    - **chunking_config**: Конфигурация поиска (опционально)
    """
    try:
        logger.info(f"Получен запрос: {request.query}")

        rag = get_rag_engine()

        # Используем конфиг из запроса или дефолтный
        chunking_cfg = request.chunking_config or config.SELECTED_C_CONFIG

        # Поиск контекста
        context = rag.search(
            query=request.query,
            chunkung_config=chunking_cfg
        )

        # Генерация ответа
        if config.G_USE_REMOTE_MODEL:
            result = rag.generator_service.ask_api(
                query=request.query,
                context=context
            )
        else:
            result = rag.generator_service.generate_response(
                query=request.query,
                context=context
            )

        # Распаковка результата
        if isinstance(result, tuple) and len(result) == 3:
            answer, input_tokens, output_tokens = result
        else:
            answer = result if isinstance(result, str) else str(result)
            input_tokens = None
            output_tokens = None

        # Подсчет чанков
        chunks_count = len(context.split('\n')) if context else 0

        return QueryResponse(
            query=request.query,
            answer=answer,
            context=context if config.G_PRINT_CHUNKS else None,
            chunks_count=chunks_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )


    except ValueError as e:
        logger.error(f"Ошибка валидации: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обработке запроса: {str(e)}"
        )


@router.post(
    "/query/stream",
    summary="Потоковый ответ от RAG-системы",
    description="Получение ответа в потоковом режиме (для длинных ответов)"
)
async def query_rag_stream(request: QueryRequest):
    """
    Потоковая генерация ответа (если поддерживается генератором)
    """
    # TODO: Реализовать потоковую генерацию при необходимости
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Потоковая генерация пока не реализована"
    )
