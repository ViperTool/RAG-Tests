import re
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

import chromadb
import torch
import torch.nn.functional as f
import numpy as np
from rank_bm25 import BM25Okapi
from scipy.ndimage import gaussian_filter
from transformers import AutoTokenizer, T5EncoderModel

from src.inference import ROWInferencer
import src.parser
import src.chroma_handler
import src.sqlite_handler


def pool(hidden_state, mask, pooling_method="cls"):
    """
    Пуллинг скрытых состояний.
    """
    if pooling_method == "mean":
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    elif pooling_method == "cls":
        return hidden_state[:, 0]
    else:
        raise ValueError(f"Unknown pooling method: {pooling_method}")


def get_embedding(text: str, tokenizer, model) -> np.ndarray:
    """
    Генерирует эмбеддинг для текста с помощью заданной модели.
    """
    tokenized = tokenizer(
        text,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**tokenized)

    embeddings = pool(
        outputs.last_hidden_state,
        tokenized["attention_mask"],
        pooling_method="cls",
    )
    embeddings = f.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


def preprocess_text(text: str) -> List[str]:
    """
    Простая предобработка текста для BM25.
    """
    tokens = re.findall(r"\w+", text.lower())
    return tokens


def initialize_embedding_model(model_name: str) -> Tuple[AutoTokenizer, T5EncoderModel]:
    """
    Загружает токенизатор и модель для эмбеддингов.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )
    model = T5EncoderModel.from_pretrained(
        model_name
    )
    return tokenizer, model


def create_bm25_index(corpus: List[str]) -> BM25Okapi:
    """
    Создаёт индекс BM25 на основе корпуса.
    """
    tokenized_corpus = [preprocess_text(doc) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def get_context_via_kernel_method(
        doc_metadatas, corpus, doc_ids_from_get, retrieved_indices, sigma=3.0, threshold=0.12
):
    """
    Применяет Kernel Density Estimation (KDE) для поиска контекста с учетом границ страниц.

    Args:
        doc_metadatas: список словарей с 'page_id' и 'chunk_order'
        corpus: список текстов
        doc_ids_from_get: список ID чанков (например, из БД)
        retrieved_indices: СПИСОК индексов (int) чанков, которые нашла нейросеть (в списке corpus)
        sigma: ширина окна внимания (аналог window, но мягче)
        threshold: порог отсечения шума

    Returns:
        list: список ID чанков (order), попавших в итоговый контекст
    """

    # 1. Предварительная группировка всего корпуса по страницам
    # { page_id: [(chunk_order, original_index), ...] }
    # Это нужно, чтобы быстро строить таймлайн для каждой страницы
    pages_map = {}
    for idx, meta in enumerate(doc_metadatas):
        pid = meta['page_id']
        order = int(meta['chunk_order'])

        if pid not in pages_map:
            pages_map[pid] = []
        pages_map[pid].append((order, idx))

    # 2. Определяем, какие страницы вообще затронул поиск
    # set найденных индексов для быстрой проверки
    retrieved_set = set(retrieved_indices)

    # Собираем ID страниц, где есть хотя бы один найденный чанк
    relevant_pages = set()
    for idx in retrieved_indices:
        relevant_pages.add(doc_metadatas[idx]['page_id'])

    final_context_objects = []

    # 3. Обрабатываем только нужные страницы
    for pid in relevant_pages:
        # Все чанки этой страницы (order, original_index)
        page_chunks = pages_map[pid]

        # Находим максимальный chunk_order на этой странице для размера массива
        max_order = max(item[0] for item in page_chunks)

        # --- KDE ЛОГИКА ---
        # Создаем таймлайн страницы
        timeline = np.zeros(max_order + 1)

        # Ставим "единички" там, где поиск нашел чанк
        for order, orig_idx in page_chunks:
            if orig_idx in retrieved_set:
                timeline[order] = 1.0

        # Размываем
        density = gaussian_filter(timeline, sigma=sigma)

        # Получаем номера (chunk_order), которые прошли порог
        passing_orders_mask = np.where(density > threshold)[0]
        passing_orders_set = set(passing_orders_mask)

        # --- СБОР РЕЗУЛЬТАТА ПО СТРАНИЦЕ ---
        for order, orig_idx in page_chunks:
            if order in passing_orders_set:
                final_context_objects.append({
                    'id': doc_ids_from_get[orig_idx],
                    'text': corpus[orig_idx],
                    'metadata': doc_metadatas[orig_idx],
                    # Для сортировки: сначала страница, потом порядок
                    'sort_key': (pid, order)
                })

    # 4. Сортируем результат (чтобы текст шел последовательно)
    # Сортировка может быть сложной, если page_id - строки.
    # Здесь сортируем по тому порядку, как они встретились бы в книге.
    final_context_objects.sort(key=lambda x: x['sort_key'])

    # 5. Возвращаем только ID (как в вашей старой функции variable 'order')
    # Если нужны сами тексты или объекты, можно вернуть final_context_objects
    return [item['id'] for item in final_context_objects]


def get_context_around_chunk(
    doc_metadatas, corpus, doc_ids_from_get, target_idx, window=2
):
    """
    Возвращает тексты чанков, метаданные и ID, находящиеся в окрестности
    заданного чанка (по индексу в списке) с тем же page_id и chunk_order в диапазоне.

    Args:
        doc_metadatas: список словарей с 'page_id' и 'chunk_order'
        corpus: список текстов
        doc_ids_from_get: список ID чанков
        target_idx: индекс целевого чанка в списке
        window: сколько чанков брать до и после (default: 2)

    Returns:
        list: список словарей {'id', 'text', 'metadata'}
    """
    target_meta = doc_metadatas[target_idx]
    target_page_id = target_meta['page_id']
    target_chunk_order = int(target_meta['chunk_order'])

    min_order = target_chunk_order - window
    max_order = target_chunk_order + window

    context = []
    order = []

    for i, meta in enumerate(doc_metadatas):
        if meta['page_id'] == target_page_id:
            chunk_order = int(meta['chunk_order'])
            if min_order <= chunk_order <= max_order:
                context.append({
                    'id': doc_ids_from_get[i],
                    'text': corpus[i],
                    'metadata': meta
                })
                order.append(doc_ids_from_get[i])

    context.sort(key=lambda x: int(x['metadata']['chunk_order']))
    return order


def lexical_search(
    query: str, bm25: BM25Okapi, corpus: List[str], doc_ids: List[str], n: int
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Выполняет лексический поиск (BM25).
    """
    tokenized_query = preprocess_text(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_n_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:n]

    results = []
    for i in top_n_indices:
        results.append({
            "id": doc_ids[i],
            "document": corpus[i],
            "score": bm25_scores[i],
        })
    print(f"Лексический поиск: {top_n_indices}")
    return results, top_n_indices


def semantic_search(
    query_embedding: np.ndarray,
    collection,
    n: int
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Выполняет семантический (векторный) поиск.
    """
    vector_results = collection.query(
        query_embeddings=query_embedding, n_results=n
    )

    vector_docs = vector_results["documents"][0]
    vector_doc_ids = vector_results["ids"][0]
    vector_distances = vector_results["distances"][0]
    vector_scores = [1 - d for d in vector_distances]

    results = []
    for doc_id, doc_text, v_score, dist in zip(
        vector_doc_ids, vector_docs, vector_scores, vector_distances
    ):
        results.append({
            "id": doc_id,
            "document": doc_text,
            "score": v_score,
            "distance": dist,
        })

    top_n_indices = [int(vec.replace('chunk_', '')) for vec in vector_doc_ids]
    print(f"Векторный поиск: {top_n_indices}")
    return results, top_n_indices


def combine_search_results(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    doc_ids_from_get: List[str],
    bm25_scores_full: List[float],
) -> List[Dict[str, Any]]:
    """
    Объединяет результаты векторного и BM25 поиска.
    """
    combined_results = {}

    for item in vector_results:
        doc_id = item["id"]
        doc_text = item["document"]
        v_score = item["score"]

        if doc_id in combined_results:
            existing = combined_results[doc_id]
            existing["vector_score"] = max(v_score, existing["vector_score"])
        else:
            try:
                doc_idx_in_corpus = doc_ids_from_get.index(doc_id)
                b_score = bm25_scores_full[doc_idx_in_corpus]
            except ValueError:
                b_score = 0.0

            combined_results[doc_id] = {
                "document": doc_text,
                "vector_score": v_score,
                "bm25_score": b_score,
            }

    for item in bm25_results:
        doc_id = item["id"]
        doc_text = item["document"]
        b_score = item["score"]

        if doc_id in combined_results:
            existing = combined_results[doc_id]
            existing["bm25_score"] = max(b_score, existing["bm25_score"])
        else:
            combined_results[doc_id] = {
                "document": doc_text,
                "vector_score": 0.0,
                "bm25_score": b_score,
            }

    return list(combined_results.values())


def calculate_hybrid_score(
    result: Dict[str, float], w_vector: float = 0.4, w_bm25: float = 0.6
) -> float:
    """
    Рассчитывает гибридный скор.
    """
    v_score = result["vector_score"]
    b_score = result["bm25_score"]
    return w_vector * v_score + w_bm25 * b_score


def rank_and_filter_results(
    combined_results: List[Dict[str, Any]], k: int
) -> List[Dict[str, Any]]:
    """
    Ранжирует объединённые результаты и возвращает топ-K.
    """
    sorted_results = sorted(
        combined_results,
        key=lambda x: calculate_hybrid_score(x),
        reverse=True,
    )
    return sorted_results[:k]


def build_prompt(context: str, query: str) -> str:
    """
    Формирует промпт для LLM.
    """
    return f"Контекст: {context}\n\nВопрос: {query}\n\nОтвет:"

@src.parser.log_execution
def ask(QUERY: str) -> str:
    """
    Основная функция для выполнения гибридного поиска и генерации ответа.
    """

    EMBEDDING_MODEL_NAME = "ai-forever/FRIDA"
    CHROMA_PATH = "src/data"
    COLLECTION_NAME = "wiki_chunks"
    N_BM25 = 5
    N_VECTOR = 5
    K_FINAL = 5

    system_prompt = "### Источник данных ###\nДля ответа используй ИСКЛЮЧИТЕЛЬНО предоставленные ниже материалы из вики-базы знаний Outer Wilds:\n\n### Строгие инструкции ###\n\n1. Анализ запроса: Выдели из вопроса ключевые элементы:\n * Название локации/планеты\n * Персонаж/раса\n * Предмет/артефакт\n * Механика/способность\n * Секрет/достижение\n\n2. Основа ответа: Каждое утверждение в ответе должно иметь прямое подтверждение в предоставленном контексте. Запрещено:\n * Придумывать местоположения предметов без подтверждения в контексте\n * Расшифровывать древние символы или языки без их объяснения в контексте\n * Предполагать последствия действий, не описанных в контексте\n * Использовать информацию о сюжете не из официальной вики\n\n3. Структура ответа:\n\n Анализ запроса:\n В вопросе упоминаются следующие ключевые элементы: [перечисли элементы из п.1]. \n В вики-базе найдена следующая релевантная информация: [кратко опиши, что именно в контексте относится к этим элементам].\n\n Доступная информация из вики:\n [Строго на основе контекста представь информацию об элементах запроса. Если в контексте есть:\n - Описания локаций → опиши их\n - Указания маршрутов → приведи их\n - Способы взаимодействия с объектами → опиши их\n - Сюжетные детали → приведи их в точном соответствии с контекстом\n Если такой информации нет, не придумывай!]\n\n Связанные элементы:\n [Если в контексте упоминаются связанные локации, персонажи или предметы, перечисли их с краткими пояснениями из вики. Если связей нет, так и укажи.]\n\n Важное замечание:\n Если информация в вики неполная или отсутствует по каким-то аспектам вопроса, прямо укажи: 'В вики-базе отсутствует информация о [конкретный аспект]'.\n\n### Важно: ###\n- Используй терминологию и названия точно в том виде, в котором они представлены в контексте\n- Не добавляй спойлеров о сюжете, если они не присутствуют в предоставленном контексте\n- Если в контексте есть противоречивая информация, сообщи об этом, указав оба варианта\n- Сохраняй спойлер-фри подход: не раскрывай ключевые сюжетные повороты, если в вопросе не запрашивается конкретно эта информация\n\n### Вопрос игрока: ###\n"
    # 1. Инициализация модели эмбеддингов
    tokenizer_emb, model_emb = initialize_embedding_model(EMBEDDING_MODEL_NAME)

    # 2. Загрузка коллекции из Chroma
    corpus, doc_metadatas, doc_ids_from_get = src.chroma_handler.load_collection(
        COLLECTION_NAME, CHROMA_PATH
    )

    # 3. Создание индекса BM25
    bm25 = create_bm25_index(corpus)

    # 4. Лексический поиск (BM25)
    bm25_results, _ = lexical_search(
        QUERY, bm25, corpus, doc_ids_from_get, N_BM25
    )

    # 5. Семантический поиск
    query_embedding = get_embedding(QUERY, tokenizer_emb, model_emb)

    # Необходимо получить объект collection снова для семантического поиска
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    vector_results, _ = semantic_search(query_embedding, collection, N_VECTOR)

    # 6. Объединение результатов
    # Для корректного объединения нужен полный список bm25_scores
    tokenized_query_full = preprocess_text(QUERY)
    full_bm25_scores = bm25.get_scores(tokenized_query_full)
    combined_results = combine_search_results(
        vector_results, bm25_results, doc_ids_from_get, full_bm25_scores
    )

    # 7. Ранжирование и фильтрация
    ranked_results = rank_and_filter_results(combined_results, K_FINAL)

    # 8. Формирование контекста и промпта
    context = "\n".join([res["document"] for res in ranked_results])
    prompt = build_prompt(context, QUERY)
    print(prompt)

    # 9. Генерация ответа
    spi = ROWInferencer()
    answer = spi.generate_response(user_prompt=prompt, system_prompt=system_prompt)
    print(answer["response"])

    return answer["response"]

@src.parser.log_execution
def ask_swr(QUERY: str) -> str:
    """
    Основная функция для выполнения гибридного поиска и генерации ответа.
    """

    EMBEDDING_MODEL_NAME = "./src/fine_tuned_frida_opt"
    # EMBEDDING_MODEL_NAME = "ai-forever/FRIDA"
    CHROMA_PATH = "src/data"
    COLLECTION_NAME = "wiki_pages_swr"
    N_BM25 = 5
    N_VECTOR = 5
    K_FINAL = 10
    FINAL_CHUNKS = []

    system_prompt = "### Источник данных ###\nДля ответа используй ИСКЛЮЧИТЕЛЬНО предоставленные ниже материалы из вики-базы знаний Outer Wilds:\n\n### Строгие инструкции ###\n\n1. Анализ запроса: Выдели из вопроса ключевые элементы:\n * Название локации/планеты\n * Персонаж/раса\n * Предмет/артефакт\n * Механика/способность\n * Секрет/достижение\n\n2. Основа ответа: Каждое утверждение в ответе должно иметь прямое подтверждение в предоставленном контексте. Запрещено:\n * Придумывать местоположения предметов без подтверждения в контексте\n * Расшифровывать древние символы или языки без их объяснения в контексте\n * Предполагать последствия действий, не описанных в контексте\n * Использовать информацию о сюжете не из официальной вики\n\n3. Структура ответа:\n\n Анализ запроса:\n В вопросе упоминаются следующие ключевые элементы: [перечисли элементы из п.1]. \n В вики-базе найдена следующая релевантная информация: [кратко опиши, что именно в контексте относится к этим элементам].\n\n Доступная информация из вики:\n [Строго на основе контекста представь информацию об элементах запроса. Если в контексте есть:\n - Описания локаций → опиши их\n - Указания маршрутов → приведи их\n - Способы взаимодействия с объектами → опиши их\n - Сюжетные детали → приведи их в точном соответствии с контекстом\n Если такой информации нет, не придумывай!]\n\n Связанные элементы:\n [Если в контексте упоминаются связанные локации, персонажи или предметы, перечисли их с краткими пояснениями из вики. Если связей нет, так и укажи.]\n\n Важное замечание:\n Если информация в вики неполная или отсутствует по каким-то аспектам вопроса, прямо укажи: 'В вики-базе отсутствует информация о [конкретный аспект]'.\n\n### Важно: ###\n- Используй терминологию и названия точно в том виде, в котором они представлены в контексте\n\n### Вопрос игрока: ###\n"
    # 1. Инициализация модели эмбеддингов
    tokenizer_emb, model_emb = initialize_embedding_model(EMBEDDING_MODEL_NAME)
    print(f"Модель: {EMBEDDING_MODEL_NAME}")
    # 2. Загрузка коллекции из Chroma
    corpus, doc_metadatas, doc_ids_from_get = src.chroma_handler.load_collection(
        COLLECTION_NAME, CHROMA_PATH
    )

    # 3. Создание индекса BM25
    bm25 = create_bm25_index(corpus)

    # 4. Лексический поиск (BM25)
    bm25_results, bm25_ids = lexical_search(
        QUERY, bm25, corpus, doc_ids_from_get, N_BM25
    )

    bm25_results = []
    bm25_ids = []

    # 5. Семантический поиск
    query_embedding = get_embedding(QUERY, tokenizer_emb, model_emb)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    vector_results, vector_ids = semantic_search(query_embedding, collection, N_VECTOR)

    # 6. Формирование итоговых чанков
    FINAL_CHUNKS = list(sorted(set(tuple(get_context_via_kernel_method(doc_metadatas, corpus, doc_ids_from_get, bm25_ids + vector_ids)))))
    print(f"Количество итоговых чанков: {len(FINAL_CHUNKS)} | Итоговые чанки: {FINAL_CHUNKS}")

    results = collection.get(ids=FINAL_CHUNKS, include=["documents"])
    orig_chunks = [res for res in results["documents"]]
    unique_chunks = list(dict.fromkeys(orig_chunks))

    print(f"Было чанков: {len(orig_chunks)}")
    print(f"Стало чанков: {len(unique_chunks)}")
    print(f"Удалено дубликатов: {len(orig_chunks) - len(unique_chunks)}")

    context = "\n".join([res for res in unique_chunks])
    prompt = build_prompt(context, QUERY)
    print(f"Полученный промпт: {prompt}")

    spi = ROWInferencer()
    answer = spi.generate_response(user_prompt=prompt, system_prompt=system_prompt)
    print(answer["response"])

    return answer["response"]


if __name__ == "__main__":
    QUERY = "Что происходит с призрачной материей?"
    ask_swr(QUERY)
    # ask(QUERY)