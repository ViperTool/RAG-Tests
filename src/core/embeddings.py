import numpy as np
import gc
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel
from typing import List, Union

from src.utils import config
from src.utils import exceptions


class EmbeddingService:
    def __init__(self):
        self.device = config.R_DEVICE
        self.tokenizer = None
        self.model = None

    def load(self, model_name: str = config.R_EMBEDDING_MODEL_NAME):
        """
        Загрузка модели
        """
        print(f"Загрузка модели {model_name} на {self.device.upper()}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def unload(self):
        """
        Принудительная выгрузка модели из VRAM
        """
        print("Выгрузка модели эмбеддингов из VRAM...")

        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
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

            pooled_embeds = self.pool(
                model_output.last_hidden_state,
                encoded_input['attention_mask']
            )

            if normalize:
                pooled_embeds = F.normalize(pooled_embeds, p=2, dim=1)

            all_embeddings.append(pooled_embeds.cpu().numpy())

        return np.vstack(all_embeddings)
