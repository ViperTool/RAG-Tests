from sentence_transformers import CrossEncoder
from typing import List, Tuple
import gc
import logging.config
from rag_core.src.utils import exceptions
import config

logger = logging.getLogger(__name__)

class CrossEncoderService:
    def __init__(self):
        self.model = None
        self.device = config.CE_DEVICE

    def load(
        self,
        model_name: str = config.CE_LOCAL_MODEL_NAME,
        max_length: int = 512
    ):
        if self.model is not None:
            logger.info("Кроссэнкодерная модель уже загружена.")
            return

        try:
            logger.info(f"Загрузка кроссэнкодерной модели {model_name} на {self.device.upper()}...")
            self.model = CrossEncoder(
                model_name,
                device=self.device,
                max_length=max_length
            )
            self.model.eval()
            logger.info("Кроссэнкодерная модель успешно загружена.")
        except OSError as e:
            logger.critical(f"Не удалось найти или скачать кроссэнкодерную модель {model_name}: {e}")
            raise exceptions.ModelLoadingError(f"Ошибка файлов кроссэнкодерной модели: {e}") from e
        except Exception as e:
            logger.critical(f"Ошибка инициализации CrossEncoderManager: {e}")
            raise exceptions.ModelLoadingError(f"Неизвестная ошибка инициализации CrossEncoderManager: {e}") from e

    def unload(self):
        logger.info("Выгрузка кроссэнкодерной модели из VRAM...")
        try:
            if self.model:
                del self.model
        except Exception as e:
            logger.warning(f"Ошибка при удалении кроссэнкодерной модели: {e}")
        finally:
            self.model = None
            gc.collect()
            logger.info("Память очищена")

    def rerank(self, query: str, documents: List[str], top_k: int) -> List[Tuple[str, float]]:
        if self.model is None:
            raise RuntimeError("Кросс-энкодер модель не загружена. Вызовите load_model() перед rerank().")

        if top_k <= 0:
            raise ValueError("top_k должно быть > 0")

        results = self.model.rank(
            query,
            documents,
            top_k=top_k,
            return_documents=False,
            show_progress_bar=False
        )

        ranked_docs = []
        for res in results:
            doc = documents[res["corpus_id"]]
            score = res["score"]
            ranked_docs.append((doc, score))

        logger.info(f"Результат реранжирования:\n{ranked_docs}")
        return [doc[0] for doc in ranked_docs]