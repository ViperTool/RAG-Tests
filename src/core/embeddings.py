import numpy as np
import gc
import logging.config
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, T5EncoderModel
from typing import List, Union

from src.utils import exceptions
import config

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.device = config.R_DEVICE
        self.tokenizer = None
        self.model = None

    def load(self, model_name: str = config.R_EMBEDDING_MODEL_NAME):
        """
        Загрузка модели
        """
        logger.info(f"Загрузка эмбеддинговой модели {model_name} на {self.device.upper()}...")

        if self.model is not None:
            logger.info("Эмбеддинговая модель уже загружена.")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = T5EncoderModel.from_pretrained(model_name)
            self.model.eval()
            self.model.to(self.device)
            logger.info("Эмбеддинговая модель успешно загружена.")
        except OSError as e:
            logger.critical(f"Файлы эмбеддинговой модели {model_name} не найдены: {e}")
            raise exceptions.EmbeddingModelLoadingError(f"Ошибка файлов эмбеддинговой модели: {e}") from e
        except torch.cuda.OutOfMemoryError as e:
            logger.critical(f"Недостаточно VRAM для загрузки эмбеддинговой модели {model_name}.")
            self.unload()
            raise exceptions.EmbeddingModelLoadingError("Недостаточно VRAM для загрузки эмбеддинговой модели") from e
        except Exception as e:
            logger.critical(f"Ошибка инициализации EmbeddingService: {e}")
            raise exceptions.EmbeddingModelLoadingError(f"Неизвестная ошибка инициализации EmbeddingService: {e}") from e

    def unload(self):
        """
        Принудительная выгрузка модели из VRAM
        """
        print("Выгрузка модели эмбеддингов из VRAM...")

        logger.info("Выгрузка эмбеддинговой модели...")
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
        except Exception as e:
            logger.warning(f"Ошибка при удалении объектов эмбеддинговой модели: {e}")
        finally:
            self.model = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Память очищена.")

    @staticmethod
    def _pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Единая логика пулинга
        """
        if config.R_POOLING_METHOD == 'cls':
            return last_hidden_state[:, 0, :]
        elif config.R_POOLING_METHOD == 'mean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        else:
            raise ValueError(f"Неизвестный метод пулинга: {config.R_POOLING_METHOD}")

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Генерирует эмбеддинги для строки или списка строк.
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        try:
            for i in range(0, len(texts), config.R_BATCH_SIZE):
                batch = texts[i: i + config.R_BATCH_SIZE]

                encoded_input = self.tokenizer(
                    batch,
                    max_length=config.R_MAX_LENGTH,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)

                with torch.no_grad():
                    model_output = self.model(**encoded_input)

                pooled_embeds = self._pool(
                    model_output.last_hidden_state,
                    encoded_input['attention_mask']
                )

                if normalize:
                    pooled_embeds = F.normalize(pooled_embeds, p=2, dim=1)

                all_embeddings.append(pooled_embeds.cpu().numpy())

            return np.vstack(all_embeddings)
        except torch.cuda.OutOfMemoryError as e:
            logger.error("Недостаточно памяти при кодировании текста.")
            torch.cuda.empty_cache()
            raise exceptions.EncodeError("Не хватило памяти для батча") from e
        except Exception as e:
            logger.error(f"Ошибка кодирования: {e}")
            raise exceptions.EncodeError(f"Ошибка при создании эмбеддингов: {e}") from e