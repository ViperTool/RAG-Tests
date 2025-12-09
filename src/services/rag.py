import config

import numpy as np
from typing import List
from rank_bm25 import BM25Okapi
from scipy.ndimage import gaussian_filter
import re

from src.core.embeddings import EmbeddingService
from src.core.generators import GeneratorService
from src.core.ner import NERService
from src.chroma.chroma_handler import get_chroma_client


class RAGEngine:
    def __init__(self, collection_name: str):
        try:
            print("Инициализация RAG-системы...")
            self.embedding_service = EmbeddingService()

            print(f"Параметр выгрузки эмбеддинг-модели после генерации (R_UNLOAD_ON_GENERATION) установлен {config.R_UNLOAD_ON_GENERATION}: ", end='')
            self.embedding_service.load() if not config.R_UNLOAD_ON_GENERATION else print(f"Модель будет загружена по необходимости.")

            print(f"Параметр загрузки удалённой модели-генератора (G_USE_REMOTE_MODEL) установлен {config.G_USE_REMOTE_MODEL}: ", end='')
            self.generator_service = GeneratorService()
            self.generator_service.load() if not config.G_USE_REMOTE_MODEL else print(f"Запрос будет выполняться сторонним сервисом моделью {config.G_REMOTE_MODEL_NAME}.")

            print("Загрузка индекса BM25...")
            self.client = get_chroma_client()
            self.collection = self.client.get_collection(name=collection_name)
            data = self.collection.get(include=["documents", "metadatas", "embeddings"])
            self.corpus = data["documents"]
            self.metadatas = data["metadatas"]
            self.doc_ids = data["ids"]
            tokenized_corpus = [self.preprocess_text(doc) for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)

            print("Загрузка NER-модели...")
            self.ner_service = NERService()


            print("RAG-система готова.")
        except Exception as e:
            print(f"Ошибка при инициализации RAG-системы: {e}")

    @staticmethod
    def preprocess_text(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, n_bm25: int = 5, n_vector: int = 20) -> str:
        """
        Главный метод: находит релевантный контекст для вопроса.
        Возвращает строку с контекстом.

        Args:
            query (str): Запрос пользователя
            n_bm25 (int): Количество возвращаемых чанков по лексическому поиску
            n_vector (int): Количество возвращаемых чанков по семантическому поиску

        TODO:
        Дополнительные способы поиска
        Добавление реранкера
        Динамическое количество чанков по лексическому и семантическому поиску
        """
        query = self.ner_service.extract_search_terms(query)
        tokenized_query = self.preprocess_text(query)
        bm25_scores_full = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores_full)[::-1][:n_bm25].tolist()

        print(f"Результаты лексического поиска: {top_bm25_indices}")

        if self.embedding_service.model is None:
            self.embedding_service.load()

        query_vec = self.embedding_service.encode(query, normalize=True)

        if config.R_UNLOAD_ON_GENERATION:
            self.embedding_service.unload()

        vector_results = self.collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=n_vector
        )

        vector_indices = []
        for vid in vector_results['ids'][0]:
            try:
                idx = self.doc_ids.index(vid)
                vector_indices.append(idx)
            except ValueError:
                continue

        print(f"Результаты векторного поиска: {vector_indices}")

        all_candidate_indices = list(set(top_bm25_indices) | set(vector_indices))
        final_indices = self.apply_kernel_method(all_candidate_indices)
        context_parts = [self.corpus[i] for i in final_indices]
        return "\n".join(context_parts)

    def answer(self, query: str, context: str):
        return self.generator_service.generate_response(query, context)

    def apply_kernel_method(self, retrieved_indices: List[int], sigma=4.0, threshold=0.13) -> List[int]:
        """
        Расширение контекста, основанное на гауссовском фильтре.
        """

        print(f"Начальные чанки: {retrieved_indices}")
        print(f"Начальные чанки (отсортированные): {sorted(retrieved_indices)}")

        pages_map = {}

        relevant_page_ids = set()
        for idx in retrieved_indices:
            meta = self.metadatas[idx]
            pid = meta['page_id']
            relevant_page_ids.add(pid)

        # В идеале карту страниц надо построить один раз в __init__, если корпус не меняется
        for idx, meta in enumerate(self.metadatas):
            pid = meta['page_id']
            if pid in relevant_page_ids:
                if pid not in pages_map:
                    pages_map[pid] = []
                pages_map[pid].append((int(meta['chunk_order']), idx))

        final_indices = retrieved_indices

        for pid in relevant_page_ids:
            page_chunks = pages_map[pid]
            if not page_chunks: continue

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

        print(f"Итоговые чанки: {total_indices}")

        return total_indices
