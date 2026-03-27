from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
import httpx
import os

from src.schemas.request_models import QueryRequest
from src.schemas.response_models import QueryResponse, ErrorResponse, HealthResponse

RAG_CORE_URL = os.getenv("RAG_CORE_URL", "http://localhost:8001")

# Создание роутера
router = APIRouter(
    prefix="/api/v1/rag",
    tags=["Public RAG"],
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        400: {"model": ErrorResponse, "description": "Bad Request"}
    }
)

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Задать вопрос RAG-системе",
    description="Отправляет запрос в RAG-систему и получает ответ на основе базы знаний"
)
async def query_public_api(request: QueryRequest):
    """Публичный эндпоинт, который пересылает запрос во внутренний RAG-сервис

        - **query**: Вопрос пользователя (обязательно)
        - **chunking_config**: Конфигурация поиска (опционально)
    """

    # Используем асинхронный HTTP-клиент httpx
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # Отправляем запрос в первый контейнер
            response = await client.post(
                f"{RAG_CORE_URL}/internal/query",
                json={
                    "query": request.query,
                    "chunking_config": request.chunking_config
                      }
            )
            response.raise_for_status()  # Проверка на ошибки HTTP (400, 500)

            data = response.json()
            return QueryResponse(
                query=request.query,
                answer=data["answer"],
                chunks_count=data["chunks_count"],
                input_tokens=data["input_tokens"],
                output_tokens=data["output_tokens"]
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Ошибка RAG сервиса: {e.response.text}"
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Внутренний RAG сервис недоступен: {str(e)}"
            )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния сервиса"
)
async def health_check():
    """Проверка работоспособности API и связи с ядром"""
    async with httpx.AsyncClient(timeout=10.0) as client:  # Таймаут меньше для health
        try:
            # Отправляем запрос в первый контейнер
            response = await client.get(f"{RAG_CORE_URL}/health")

            # Если rag_core вернул 503 (unhealthy), это не исключение для нас,
            # мы просто должны пробросить этот статус дальше
            if response.status_code == 503:
                data = response.json()
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy (RAG core issue)",
                        "error": data.get("error", "Unknown error in core")
                    }
                )

            response.raise_for_status()
            data = response.json()

            return HealthResponse(
                status=data.get("status", "unknown"),
                models_loaded=data.get("models_loaded", {})
            )

        except httpx.ConnectError:
            # RAG-ядро вообще недоступно (выключено)
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": "RAG core is offline or booting up"
                }
            )
        except httpx.TimeoutException:
            # RAG-ядро зависло
            return JSONResponse(
                status_code=504,
                content={
                    "status": "unhealthy",
                    "error": "RAG core timeout (is it loading models?)"
                }
            )
        except Exception as e:
            # Любая другая ошибка в web_api
            return JSONResponse(
                status_code=500,
                content={"status": "error", "error": f"Web API internal error: {str(e)}"}
            )
