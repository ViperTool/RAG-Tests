from pydantic import BaseModel, Field
from typing import Optional


class QueryResponse(BaseModel):
    """Модель ответа RAG-системы"""
    query: str = Field(..., description="Исходный запрос пользователя")
    answer: str = Field(..., description="Сгенерированный ответ")
    chunks_count: Optional[int] = Field(None, description="Количество использованных чанков")
    input_tokens: Optional[int] = Field(None, description="Количество входных токенов")
    output_tokens: Optional[int] = Field(None, description="Количество выходных токенов")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Какие планеты есть в солнечной системе?",
                "answer": "В солнечной системе Outer Wilds есть следующие планеты...",
                "chunks_count": 5,
                "input_tokens": 2029,
                "output_tokens": 482
            }
        }


class HealthResponse(BaseModel):
    """Статус работы сервиса"""
    status: str = Field(..., description="Статус сервиса")
    models_loaded: dict = Field(..., description="Загруженные модели")


class ErrorResponse(BaseModel):
    """Модель ошибки"""
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Детали ошибки")
