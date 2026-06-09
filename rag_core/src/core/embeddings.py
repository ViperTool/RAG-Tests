import numpy as np
import logging
import requests
from typing import List, Union

from src.utils import exceptions
import config

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        # Предполагается, что в config появится URL для API эмбеддингов
        self.base_url = getattr(config, 'R_EMBEDDING_API_URL', 'http://localhost:8080/v1')
        self.model_name = config.R_EMBEDDING_MODEL_NAME

    def load(self, model_name: str = None):
        """
        Заглушка для обратной совместимости.
        Модель загружается на стороне сервера, поэтому здесь ничего не делаем.
        """
        logger.info("Используется API эмбеддингов. Загрузка весов не требуется.")

    def unload(self):
        """
        Заглушка для обратной совместимости.
        """
        logger.info("Используется API эмбеддингов. Очистка VRAM управляется сервером.")

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Генерирует эмбеддинги через API llama.cpp.
        """
        if isinstance(texts, str):
            texts = [texts]

        url = f"{self.base_url}/embeddings"
        payload = {
            "input": texts,
            "model": self.model_name
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()

            data = response.json()
            # Извлекаем эмбеддинги и сортируем по индексу, чтобы сохранить порядок
            embeddings = [item['embedding'] for item in sorted(data['data'], key=lambda x: x['index'])]

            result_array = np.array(embeddings)

            # Llama.cpp обычно возвращает уже нормализованные векторы,
            # но если нужно принудительно:
            if normalize:
                norms = np.linalg.norm(result_array, axis=1, keepdims=True)
                result_array = result_array / np.where(norms == 0, 1, norms)

            return result_array

        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддингов по API: {e}")
            raise exceptions.EncodeError(f"Ошибка API эмбеддингов: {e}") from e