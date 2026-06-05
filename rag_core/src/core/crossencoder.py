import logging
import requests
from typing import List, Tuple

from src.utils import exceptions
import config

logger = logging.getLogger(__name__)


class CrossEncoderService:
    def __init__(self):
        # Аналогично, добавляем URL в конфиг
        self.base_url = getattr(config, 'CE_RERANKER_API_URL', 'http://localhost:8080/v1')
        self.model_name = config.CE_LOCAL_MODEL_NAME

    def load(self, model_name: str = None, max_length: int = 512):
        """Заглушка. Модель живет на сервере."""
        logger.info("Используется API реранкера. Загрузка весов не требуется.")

    def unload(self):
        """Заглушка."""
        pass

    def rerank(self, query: str, documents: List[str], top_k: int) -> List[str]:
        if top_k <= 0:
            raise ValueError("top_k должно быть > 0")

        url = f"{self.base_url}/rerank"
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_k
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()
            # Извлекаем отсортированные результаты
            results = data.get('results', [])

            ranked_docs = []
            for res in results:
                doc_index = res["index"]
                score = res["relevance_score"]
                ranked_docs.append((documents[doc_index], score))

            logger.info(f"Результат реранжирования (top {top_k}): {[score for _, score in ranked_docs]}")
            return [doc[0] for doc in ranked_docs]

        except Exception as e:
            logger.error(f"Ошибка при обращении к API реранкера: {e}")
            # Возвращаем исходные документы (обрезанные до top_k) как фоллбэк
            return documents[:top_k]