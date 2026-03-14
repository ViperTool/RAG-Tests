from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class QueryRequest(BaseModel):
    """Модель запроса к RAG-системе"""
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Вопрос пользователя",
        examples=["Как работает перемещение во времени в игре?"]
    )

    chunking_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Конфигурация чанкирования (если не указана, используется SELECTED_C_CONFIG)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Какие планеты есть в солнечной системе?",
                "chunking_config": None
            }
        }
