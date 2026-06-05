from src.utils import exceptions
import config

import numpy as np
import re
import logging.config
from typing import List
from rank_bm25 import BM25Okapi
from scipy.ndimage import gaussian_filter
from sklearn.metrics.pairwise import cosine_similarity

from src.core.embeddings import EmbeddingService
from src.core.generators import GeneratorService
from src.core.crossencoder import CrossEncoderService
from src.core.ner import NERService
from src.chroma.chroma_handler import ChromaManager

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, collection_name: str):
        logger.info("Инициализация RAG-системы...")

        self.embedding_service = EmbeddingService()
        logger.info(
            f"Параметр выгрузки эмбеддинг-модели после генерации (R_UNLOAD_ON_GENERATION) установлен {config.R_UNLOAD_ON_GENERATION}")

        if not config.R_UNLOAD_ON_GENERATION:
            try:
                self.embedding_service.load()
            except (FileNotFoundError, IOError) as e:
                logger.error(f"Ошибка загрузки эмбеддинг-модели: {e}")
                raise exceptions.InitializationError(f"Не удалось загрузить эмбеддинг-модель: {e}") from e
        else:
            logger.info("Модель будет загружена по необходимости.")

        logger.info(
            f"Параметр загрузки удалённой модели-генератора (G_USE_REMOTE_MODEL) установлен {config.G_USE_REMOTE_MODEL}")
        self.generator_service = GeneratorService()

        if not config.G_USE_REMOTE_MODEL:
            try:
                self.generator_service.load()
            except (FileNotFoundError, IOError, RuntimeError) as e:
                logger.error(f"Ошибка загрузки модели-генератора: {e}")
                raise exceptions.InitializationError(f"Не удалось загрузить модель-генератор: {e}") from e
        else:
            logger.info(f"Запрос будет выполняться сторонним сервисом моделью {config.G_REMOTE_MODEL_NAME}.")

        logger.info("Загрузка индекса BM25...")
        try:
            self.chroma_manager = ChromaManager()
            self.client = self.chroma_manager.client
            collections = self.chroma_manager.client.list_collections()
            for collection in collections:
                logger.info(collection.name)
            self.collection = self.client.get_collection(name=collection_name)
        except ValueError as e:
            logger.error(f"Коллекция '{collection_name}' не найдена: {e}")
            raise exceptions.InitializationError(f"Коллекция '{collection_name}' не существует") from e
        except ConnectionError as e:
            logger.error(f"Ошибка подключения к ChromaDB: {e}")
            raise exceptions.InitializationError("Не удалось подключиться к ChromaDB") from e

        try:
            data = self.collection.get(include=["documents", "metadatas", "embeddings"])
            self.corpus = data["documents"]
            self.metadatas = data["metadatas"]
            self.doc_ids = data["ids"]
            self.embeddings = np.array(data["embeddings"])
        except KeyError as e:
            logger.error(f"Отсутствуют необходимые данные в коллекции: {e}")
            raise exceptions.InitializationError(f"Некорректная структура данных в коллекции: {e}") from e

        tokenized_corpus = [self._preprocess_text(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.pages_map = self._build_pages_map()

        logger.info("Загрузка NER-модели...")
        self.ner_service = NERService()

        logger.info("Загрузка кроссэнкодер-модели...")
        self.crossencoder_service = CrossEncoderService()

        logger.info("RAG-система готова.")

    def _build_pages_map(self) -> dict:
        """
        Строит индекс {page_id: [(chunk_order, global_index), ...]} один раз при запуске.
        """
        logger.debug("Построение карты страниц для контекстного фильтра...")
        mapping = {}
        for idx, meta in enumerate(self.metadatas):
            try:
                pid = meta.get('page_id')
                order = meta.get('chunk_order')

                if pid is None or order is None:
                    continue

                if pid not in mapping:
                    mapping[pid] = []
                mapping[pid].append((int(order), idx))
            except (ValueError, TypeError):
                continue
        return mapping

    @staticmethod
    def _preprocess_text(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    from typing import Dict, Any

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """
        Проверка корректности введённых данных в конфиге чанкирования.

        Args:
            config (Dict[str, Any]): Конфиг, требующий валидации.

        Raises:
            KeyError: Если отсутствует обязательный ключ.
            TypeError: Если значение имеет неправильный тип.
            ValueError: Если значение находится вне допустимого диапазона.
        """
        required_keys = [
            "C_BM25_AMOUNT", "C_VECTOR_AMOUNT", "C_USE_RERANKER", "C_RERANKER_AMOUNT", # Общие параметры
            "C_USE_KERNEL", "C_KERNEL_SIGMA", "C_KERNEL_THRESHOLD", # Параметры метода с гауссовским фильтром
            "C_USE_DYNAMIC_EXPANSION", "C_DYNAMIC_THRESHOLD", "C_DYNAMIC_PENALTY" # Параметроы метода с широким окном
        ]

        for key in required_keys:
            if key not in config:
                raise KeyError(f"Отсутствует обязательный ключ в конфиге: {key}")

        if not isinstance(config["C_BM25_AMOUNT"], int) or config["C_BM25_AMOUNT"] < 0:
            raise ValueError("C_BM25_AMOUNT должен быть неотрицательным целым числом")
        if not isinstance(config["C_VECTOR_AMOUNT"], int) or config["C_VECTOR_AMOUNT"] < 0:
            raise ValueError("C_VECTOR_AMOUNT должен быть неотрицательным целым числом")
        if not isinstance(config["C_CONCATENATE_CHUNKS_OF_PAGES"], bool):
            raise TypeError("C_CONCATENATE_CHUNKS_OF_PAGES должен быть типа bool")
        if not isinstance(config["C_USE_RERANKER"], bool):
            raise TypeError("C_USE_RERANKER должен быть типа bool")
        if config["C_RERANKER_AMOUNT"] is not None and (
                not isinstance(config["C_RERANKER_AMOUNT"], int) or config["C_RERANKER_AMOUNT"] < 0):
            raise ValueError("C_RERANKER_AMOUNT должен быть неотрицательным целым числом или None")

        if not isinstance(config["C_USE_KERNEL"], bool):
            raise TypeError("C_USE_KERNEL должен быть типа bool")
        if config["C_KERNEL_SIGMA"] is not None and (
                not isinstance(config["C_KERNEL_SIGMA"], (int, float)) or config["C_KERNEL_SIGMA"] <= 0):
            raise ValueError("C_KERNEL_SIGMA должен быть положительным числом или None")
        if config["C_KERNEL_THRESHOLD"] is not None and (
                not isinstance(config["C_KERNEL_THRESHOLD"], (int, float)) or config["C_KERNEL_THRESHOLD"] < 0):
            raise ValueError("C_KERNEL_THRESHOLD должен быть неотрицательным числом или None")

        if not isinstance(config["C_USE_DYNAMIC_EXPANSION"], bool):
            raise TypeError("C_USE_DYNAMIC_EXPANSION должен быть типа bool")
        if config["C_DYNAMIC_THRESHOLD"] is not None and (
                not isinstance(config["C_DYNAMIC_THRESHOLD"], (int, float)) or config["C_DYNAMIC_THRESHOLD"] <= 0):
            raise ValueError("C_DYNAMIC_THRESHOLD должен быть положительным числом или None")
        if config["C_DYNAMIC_PENALTY"] is not None and (
                not isinstance(config["C_DYNAMIC_PENALTY"], (int, float)) or config["C_DYNAMIC_PENALTY"] < 0):
            raise ValueError("C_DYNAMIC_PENALTY должен быть неотрицательным числом или None")


    def search(self, query: str, chunkung_config: Dict[str, Any]) -> str:
        """
        Главный метод: находит релевантный контекст для вопроса.
        Возвращает строку с контекстом.

        Args:
            query (str): Запрос пользователя.
            chunkung_config (Dict[str, Any]): Конфиг, выбранный в config.py (Можно задать вручную).
            Ожидаемые ключи:
                - C_BM25_AMOUNT (int): Количество чанков, возвращаемых BM25.
                - C_VECTOR_AMOUNT (int): Количество чанков, возвращаемых векторным поиском.
                - C_CONCATENATE_CHUNKS_OF_PAGES (bool): Склеивать ли чанки с одной страницы в строку с одним заголовком
                - C_USE_RERANKER (bool): Использовать реранкер.
                - C_RERANKER_NAME (None | str): Название модели реранкера (если используется).
                - C_RERANKER_AMOUNT (None | int): Количество чанков для реранкера.

                - C_USE_KERNEL (bool): Использовать kernel-метод для расширения контекста.
                - C_KERNEL_SIGMA (None | float): Параметр sigma для kernel-метода.
                - C_KERNEL_THRESHOLD (None | float): Параметр threshold для kernel-метода.

                - C_USE_DYNAMIC_EXPANSION (bool): Использовать метод динамического окна.
                - C_DYNAMIC_THRESHOLD (None | int | float): Порог для динамического окна.
                - C_DYNAMIC_PENALTY (None | int | float): Пенальти к порогу для последующих чанков.

        Returns:
            str: Контекст, собранный для запроса.

        TODO:
        Дополнительные способы поиска
        """

        logger.info(f"Запрос пользователя: {query}")
        # Загрузка конфига
        self._validate_config(chunkung_config)

        # Получение ключевых слов и BM25 поиск
        key_query = self.ner_service.extract_search_terms(query)
        key_tokenized_query = self._preprocess_text(key_query)
        bm25_scores_full = self.bm25.get_scores(key_tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores_full)[::-1][:chunkung_config["C_BM25_AMOUNT"]].tolist()

        logger.info(f"Результаты лексического поиска: {top_bm25_indices}")

        # Векторный поиск
        query_vec = self.embedding_service.encode(query, normalize=True)

        vector_results = self.collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=chunkung_config["C_VECTOR_AMOUNT"]
        )

        vector_indices = []
        for vid in vector_results['ids'][0]:
            try:
                idx = self.doc_ids.index(vid)
                vector_indices.append(idx)
            except ValueError:
                continue

        logger.info(f"Результаты векторного поиска: {vector_indices}")
        # Формирование ответа
        # Сохранение уникальных значений
        all_candidate_indices = list(set(top_bm25_indices) | set(vector_indices))

        # --- Гауссовский метод ---
        if chunkung_config["C_USE_KERNEL"]:
            all_candidate_indices = self._apply_kernel_method(all_candidate_indices,
                                                              sigma=chunkung_config["C_KERNEL_SIGMA"],
                                                              threshold=chunkung_config["C_KERNEL_THRESHOLD"])

        # --- Метод динамического окна ---
        if chunkung_config["C_USE_DYNAMIC_EXPANSION"]:
            all_candidate_indices = self._apply_dynamic_expansion(
                all_candidate_indices,
                base_threshold=chunkung_config.get("C_DYNAMIC_THRESHOLD", 0.6),
                penalty_factor=chunkung_config.get("C_DYNAMIC_PENALTY", 0.05),
                max_steps=chunkung_config.get("C_DYNAMIC_MAX_STEPS", 3)
            )

        context_parts = [self.corpus[i] for i in all_candidate_indices]

        if chunkung_config["C_USE_RERANKER"]:
            self.crossencoder_service.load()
            context_parts = self.crossencoder_service.rerank(query, context_parts, chunkung_config["C_RERANKER_AMOUNT"])

        context_parts.sort()
        final_text = self._format_chunks(context_parts)
        logger.info(f"Полученный контекст:\n{final_text}")
        return final_text

    @staticmethod
    def _format_chunks(chunks: list[str]) -> str:
        """
        Функция склеивания текста по страницам для удаления дубликатов.
        """
        if not chunks:
            return ""

        result_parts = []

        first_clean = chunks[0].replace("search_document: ", "").strip()
        result_parts.append(first_clean)

        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]

            common_len = 0
            min_len = min(len(prev), len(curr))
            for j in range(min_len):
                if prev[j] == curr[j]:
                    common_len += 1
                else:
                    break

            unique_text = curr[common_len:].strip()

            if unique_text:
                pass

        final_text = chunks[0].replace("search_document: ", "").strip()

        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]

            common_len = 0
            min_len = min(len(prev), len(curr))
            for j in range(min_len):
                if prev[j] == curr[j]:
                    common_len += 1
                else:
                    break

            unique_text = curr[common_len:].strip()

            if not unique_text:
                continue

            if common_len > 17:
                final_text += " " + unique_text
            else:
                final_text += "\n" + unique_text.replace("search_document: ", "").strip()

        return final_text

    def _apply_kernel_method(self, retrieved_indices: List[int], sigma=4.0, threshold=0.13) -> List[int]:
        """
        Расширение контекста, основанное на гауссовском фильтре.
        """

        logger.info(f"Начальные чанки: {retrieved_indices}")
        logger.info(f"Начальные чанки (отсортированные): {sorted(retrieved_indices)}")

        pages_map = self.pages_map

        relevant_page_ids = set()

        for idx in retrieved_indices:
            meta = self.metadatas[idx]
            pid = meta['page_id']
            relevant_page_ids.add(pid)

        final_indices = retrieved_indices

        for pid in relevant_page_ids:
            page_chunks = pages_map[pid]
            if not page_chunks:
                continue

            max_order = max(c[0] for c in page_chunks)
            timeline = np.zeros(max_order + 1)

            retrieved_set = set(retrieved_indices)
            for order, orig_idx in page_chunks:
                if orig_idx in retrieved_set:
                    timeline[order] = 1.0

            density = gaussian_filter(timeline, sigma=sigma)

            passing_orders = np.where(density > threshold)[0]
            passing_set = set(passing_orders)

            for order, orig_idx in page_chunks:
                if order in passing_set:
                    final_indices.append(orig_idx)

        total_indices = sorted(list(set(final_indices)))

        logger.info(f"Итоговые чанки: {total_indices}")

        return total_indices

    def _apply_dynamic_expansion(self, retrieved_indices: List[int], base_threshold: float = 0.6,
                                 penalty_factor: float = 0.05, max_steps: int = 3) -> List[int]:
        """
        Динамическое расширение контекста на основе семантической близости соседей.

        Args:
            retrieved_indices: Список индексов изначально найденных чанков.
            base_threshold: Базовый порог косинусного сходства для добавления соседа.
            penalty_factor: На сколько увеличивается порог (или уменьшается скор) с каждым шагом удаления.
            max_steps: Максимальное количество шагов влево/вправо от оригинального чанка.
        """
        logger.info(f"Запуск динамического расширения. Исходные: {retrieved_indices}")

        final_indices = set(retrieved_indices)
        processed_indices = set(retrieved_indices)
        pid_order_map = {}
        for pid, chunk_list in self.pages_map.items():
            pid_order_map[pid] = {order: idx for order, idx in chunk_list}

        for idx in retrieved_indices:
            meta = self.metadatas[idx]
            pid = meta.get('page_id')
            current_order = meta.get('chunk_order')

            if pid is None or current_order is None:
                continue

            # Эмбеддинг центрального чанка (anchor)
            anchor_emb = self.embeddings[idx].reshape(1, -1)

            # Проверяем соседей в двух направлениях: -1 (влево) и +1 (вправо)
            for direction in [-1, 1]:
                for step in range(1, max_steps + 1):
                    target_order = int(current_order) + (step * direction)

                    # Проверяем, существует ли сосед с таким порядковым номером на этой странице
                    if pid in pid_order_map and target_order in pid_order_map[pid]:
                        neighbor_idx = pid_order_map[pid][target_order]

                        # Если сосед уже в списке выборки, идем дальше, но не прерываем цепочку
                        # (возможно, следующий за ним тоже релевантен, хотя это редкость)
                        if neighbor_idx in processed_indices:
                            continue

                        # Считаем близость
                        neighbor_emb = self.embeddings[neighbor_idx].reshape(1, -1)
                        similarity = cosine_similarity(anchor_emb, neighbor_emb)[0][0]

                        # Динамический порог: чем дальше от центра, тем строже требование
                        # Пример: порог 0.6, шаг 1 -> надо побить 0.6. Шаг 2 -> надо побить 0.6 + 0.05 = 0.65
                        current_threshold = base_threshold + (penalty_factor * (step - 1))

                        if similarity >= current_threshold:
                            final_indices.add(neighbor_idx)
                            logger.debug(f"Добавлен сосед {neighbor_idx} (sim={similarity:.4f} >= {current_threshold})")
                        else:
                            # Если цепочка прервалась (сосед не подошел), дальше в этом направлении не идем
                            break
                    else:
                        # Если чанка с таким номером нет (конец/начало документа), прерываем
                        break

        total_indices = sorted(list(final_indices))
        logger.info(f"Итоговые чанки после динамического расширения: {total_indices}")
        return total_indices