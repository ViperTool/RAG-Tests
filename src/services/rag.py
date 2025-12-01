import config

import numpy as np
from typing import List
from rank_bm25 import BM25Okapi
from scipy.ndimage import gaussian_filter
import re

from src.core.embeddings import T5EmbeddingService
from src.chroma.chroma_handler import get_chroma_client


class RAGEngine:
    def __init__(self, collection_name: str):
        print("Инициализация RAG-системы...")
        self.embedding_service = T5EmbeddingService()
        self.embedding_service.load()

        self.client = get_chroma_client()
        self.collection = self.client.get_collection(name=collection_name)

        print("Загрузка индекса BM25...")
        data = self.collection.get(include=["documents", "metadatas", "embeddings"])
        self.corpus = data["documents"]
        self.metadatas = data["metadatas"]
        self.doc_ids = data["ids"]

        # Строим индекс BM25
        tokenized_corpus = [self.preprocess_text(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("RAG-система готова.")

    @staticmethod
    def preprocess_text(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, n_bm25: int = 5, n_vector: int = 5) -> str:
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
        tokenized_query = self.preprocess_text(query)
        bm25_scores_full = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores_full)[::-1][:n_bm25]

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

        all_candidate_indices = list(set(top_bm25_indices) | set(vector_indices))
        final_indices = self.apply_kernel_method(all_candidate_indices)
        context_parts = [self.corpus[i] for i in final_indices]
        return "\n".join(context_parts)

    def apply_kernel_method(self, retrieved_indices: List[int], sigma=3.0, threshold=0.12) -> List[int]:
        """
        Умное расширение контекста, основанное на гауссовском фильтре.
        """
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

        final_indices = []

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

        return sorted(list(set(final_indices)))
