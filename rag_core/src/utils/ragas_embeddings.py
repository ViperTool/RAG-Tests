import logging
import numpy as np
from typing import List, Union
from langchain_core.embeddings import Embeddings
from src.core.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class RagasEmbeddingsAdapter(Embeddings):
    """
    Адаптер эмбеддингов для RAGAS, использующий существующий EmbeddingService.

    Это позволяет использовать llama.cpp API для метрик RAGAS,
    которые требуют embeddings (answer_relevancy, context_recall).
    """

    def __init__(self, embedding_service: EmbeddingService = None):
        """
        Args:
            embedding_service: Существующий сервис эмбеддингов.
                              Если None, создаст новый.
        """
        if embedding_service is None:
            logger.info("Создание нового EmbeddingService для RAGAS")
            self.service = EmbeddingService()
        else:
            logger.info("Использование существующего EmbeddingService для RAGAS")
            self.service = embedding_service

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        LangChain интерфейс: эмбеддинги для списка документов.
        """
        if not texts:
            return []

        try:
            # EmbeddingService возвращает numpy array
            embeddings = self.service.encode(texts, normalize=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Ошибка при эмбеддинге документов: {e}")
            # Возвращаем нулевые векторы как фоллбэк
            return [[0.0] * 768] * len(texts)  # FRIDA имеет размерность 768

    def embed_query(self, text: str) -> List[float]:
        """
        LangChain интерфейс: эмбеддинг для одного запроса.
        """
        try:
            embeddings = self.service.encode([text], normalize=True)
            return embeddings[0].tolist()
        except Exception as e:
            logger.error(f"Ошибка при эмбеддинге запроса: {e}")
            return [0.0] * 768  # Фоллбэк