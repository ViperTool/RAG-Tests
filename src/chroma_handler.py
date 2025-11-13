import chromadb
from chromadb import Embeddings, EmbeddingFunction, Documents
from chromadb.utils import batch_utils
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel

import src.sqlite_handler
import src.viewer
from tqdm import tqdm


class T5EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "ai-forever/FRIDA",
                 max_length: int = 512, batch_size: int = 16,
                 pooling_method: str = "cls"):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling_method = pooling_method

        # Загружаем модель и токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def _pool(self, last_hidden_state, attention_mask, pooling_method: str):
        """Функция пулинга (адаптируй под свою реализацию)"""
        if pooling_method == 'cls':
            return last_hidden_state[:, 0, :]
        elif pooling_method == 'mean':
            # Mean pooling с учетом mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")

    def __call__(self, input: Documents) -> Embeddings:
        all_embeddings = []

        for i in tqdm(range(0, len(input), self.batch_size)):
            batch_chunks = input[i:i + self.batch_size]

            # Токенизация
            tokenized_inputs = self.tokenizer(
                batch_chunks,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            # Передаём на GPU, если доступен
            if torch.cuda.is_available():
                tokenized_inputs = {k: v.cuda() for k, v in tokenized_inputs.items()}

            with torch.no_grad():
                outputs = self.model(**tokenized_inputs)

            # Применяем пулинг
            pooled_output = self._pool(
                outputs.last_hidden_state,
                tokenized_inputs["attention_mask"],
                pooling_method=self.pooling_method
            )

            # Нормализация L2
            embeddings_batch = F.normalize(pooled_output, p=2, dim=1)

            # Конвертируем в Python list
            embeddings_batch_list = embeddings_batch.cpu().numpy().tolist()
            all_embeddings.extend(embeddings_batch_list)

        return all_embeddings


def replace_chroma_ids(collection_name: str, new_ids: List[str], db_path: str = "./data"):
    """
    Заменяет ID документов в коллекции ChromaDB на новые, переданные в new_ids.
    Это делается через пересоздание коллекции.

    Args:
        collection_name: Название коллекции.
        new_ids: Новые ID для документов. Должен совпадать по длине с количеством документов.
        db_path: Путь к базе ChromaDB.
    """
    client = chromadb.PersistentClient(path=db_path)

    # Получаем старую коллекцию
    try:
        old_collection = client.get_collection(collection_name)
    except ValueError:
        print(f"Коллекция '{collection_name}' не найдена.")
        return

    # Загружаем все данные
    all_docs_result = old_collection.get(include=["documents", "metadatas", "embeddings"])

    documents = all_docs_result["documents"]
    metadatas = all_docs_result.get("metadatas", [{}] * len(documents))
    embeddings = all_docs_result.get("embeddings", None)

    if len(new_ids) != len(documents):
        print(f"Количество новых ID ({len(new_ids)}) не совпадает с количеством документов ({len(documents)}).")
        return

    # Удаляем старую коллекцию
    client.delete_collection(collection_name)
    print(f"Старая коллекция '{collection_name}' удалена.")

    # Создаём новую коллекцию
    # Предполагаем, что у вас есть та же функция эмбеддингов, что и раньше
    t5_ef = T5EmbeddingFunction(
        model_name="ai-forever/FRIDA",
        max_length=512,
        batch_size=16,
        pooling_method="cls"
    )

    new_collection = client.create_collection(name=collection_name, embedding_function=t5_ef)

    # Добавляем данные с новыми ID
    batch_size = 1000  # Для безопасности при большом объёме

    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        print(f"Добавляем батч {i // batch_size + 1}: элементы {i}-{end_idx - 1}")

        new_collection.add(
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=new_ids[i:end_idx],
            embeddings=embeddings[i:end_idx] if embeddings else None
        )

    print(f"Коллекция '{collection_name}' успешно пересоздана с новыми ID.")


def create_vector_swr(collection_name: str, chunks: List[str], page_ids: List[int],
                      chunk_orders: List[int], overwrite_collection: bool = False):
    """
    Создаёт или пересоздаёт векторную базу данных ChromaDB с метаданными.

    Args:
        collection_name (str): Название коллекции.
        chunks (List[str]): Список текстовых чанков.
        page_ids (List[int]): Список ID страниц, откуда были взяты чанки.
        chunk_orders (List[int]): Список порядковых номеров чанков (например, 0, 1, 2... на странице).
        overwrite_collection (bool): Если True, удалит существующую коллекцию перед созданием новой.
    """
    client = chromadb.PersistentClient(path="./data")

    t5_ef = T5EmbeddingFunction(
        model_name="ai-forever/FRIDA",
        max_length=512,
        batch_size=16,
        pooling_method="cls"
    )

    if overwrite_collection:
        existing_collections = [col.name for col in client.list_collections()]
        if collection_name in existing_collections:
            print(f"Коллекция '{collection_name}' существует. Удаляем её...")
            client.delete_collection(name=collection_name)
            print(f"Коллекция '{collection_name}' удалена.")

    # Создаём коллекцию
    collection = client.create_collection(name=collection_name, embedding_function=t5_ef)

    # Подготовка метаданных
    metadatas = [
        {"page_id": str(page_id), "chunk_order": str(chunk_order)}
        for page_id, chunk_order in zip(page_ids, chunk_orders)
    ]

    # IDs для документов в ChromaDB
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # 🔍 ДЕБАГ: Проверяем данные перед добавлением
    print("Проверка данных перед добавлением:")
    print(f"Количество chunks: {len(chunks)}")
    print(f"Количество metadatas: {len(metadatas)}")
    print(f"Количество ids: {len(ids)}")

    # Проверяем первые несколько элементов
    for i in range(min(3, len(chunks))):
        print(f"Элемент {i}:")
        print(f"  chunk: {type(chunks[i])} -> {chunks[i][:100]}...")
        print(f"  metadata: {metadatas[i]}")
        print(f"  id: {ids[i]}")

    # Добавление данных в коллекцию с батчингом
    batch_size = 1000  # Размер батча для избежания ошибки размера

    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        print(f"Добавляем батч {i // batch_size + 1}: элементы {i}-{end_idx - 1}")

        collection.add(
            documents=chunks[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx],
        )

    print(f"Коллекция '{collection_name}' успешно создана с {len(chunks)} элементами, включая метаданные.")

def create_vector(collection_name: str, embeddings: List[np.ndarray], chunks: List[str], overwrite_collection: bool = False):
    """
    Создаёт или пересоздаёт векторную базу данных ChromaDB.

    Args:
        collection_name (str): Название коллекции.
        embeddings (List[np.ndarray]): Список эмбеддингов для чанков.
        chunks (List[str]): Список текстовых чанков.
        overwrite_collection (bool): Если True, удалит существующую коллекцию перед созданием новой.
                                     Если False и коллекция существует, будет ошибка или поведение ChromaDB по умолчанию.
    """
    client = chromadb.PersistentClient(path="./data")
    collection_name = collection_name

    if overwrite_collection:
        existing_collections = [col.name for col in client.list_collections()]
        if collection_name in existing_collections:
            print(f"Коллекция '{collection_name}' существует. Удаляем её...")
            client.delete_collection(collection_name)
            print(f"Коллекция '{collection_name}' удалена.")
        else:
            print(f"Коллекция '{collection_name}' не существует, создаём новую.")

    collection = client.create_collection(collection_name)

    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=[f"{i}" for i in range(len(chunks))]
    )
    print(f"Коллекция '{collection_name}' успешно создана с {len(chunks)} элементами.")


def remove_duplicate(collection_name: str):
    """
    Удаляет дубликаты из коллекции ChromaDB на основе содержимого документов (documents).

    Args:
        collection_name (str): Имя коллекции.
    """
    # 1. Подключаемся к клиенту
    client = chromadb.PersistentClient(path="./data")
    collection = client.get_collection(collection_name)

    # 2. Получаем все документы, ID и (опционально) метаданные/эмбеддинги
    # 'ids' возвращаются всегда, не нужно указывать в include
    print(f"Получение всех документов из коллекции '{collection_name}'...")
    all_data = collection.get(include=['documents', 'metadatas', 'embeddings'])

    if not all_data['ids']:
        print("Коллекция пуста.")
        return

    ids = all_data['ids']
    documents = all_data['documents']
    # metadatas = all_data['metadatas'] # Не используется в этом примере
    # embeddings = all_data['embeddings'] # Не используется в этом примере

    print(f"Найдено {len(ids)} записей.")

    # 3. Определяем дубликаты на основе содержимого документов
    seen_documents = set()
    duplicate_ids = []
    unique_ids_to_keep = []

    for doc_id, doc_content in zip(ids, documents):
        if doc_content in seen_documents:
            duplicate_ids.append(doc_id)
        else:
            seen_documents.add(doc_content)
            unique_ids_to_keep.append(doc_id)

    print(f"Найдено {len(duplicate_ids)} дубликатов для удаления.")
    print(f"Останется {len(unique_ids_to_keep)} уникальных записей.")

    if not duplicate_ids:
        print("Дубликатов не найдено.")
        return

    # 4. Удаляем дубликаты
    print(f"Удаление {len(duplicate_ids)} дубликатов...")
    try:
        collection.delete(ids=duplicate_ids)
        print("Дубликаты успешно удалены.")
    except Exception as e:
        print(f"Ошибка при удалении дубликатов: {e}")
        return

    # 5. Опционально: Проверка
    remaining_data = collection.get(include=['ids']) # 'ids' всегда возвращаются
    print(f"После удаления в коллекции {len(remaining_data['ids'])} записей.")


def load_collection(collection_name: str, db_path: str = "./data") -> Tuple[List[str], List[Dict], List[str]]:
    """
    Подключается к Chroma, загружает документы, метаданные и ID.
    """
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)
    all_docs_result = collection.get(include=["documents", "metadatas"])
    corpus = all_docs_result["documents"]
    doc_metadatas = all_docs_result.get("metadatas", [{}] * len(corpus))
    doc_ids = all_docs_result["ids"]
    return corpus, doc_metadatas, doc_ids


def delete_collection(collection_name: str, db_path: str = "./data"):
    agreed = str(input(f"Вы уверены, что хотите удалить коллекцию {collection_name}? [Y/_]: "))
    if agreed.lower() == "y":
        client = chromadb.PersistentClient(path=db_path)
        client.delete_collection(collection_name)
        print("Коллекция удалена.")



def make_collection(collection_name: str):
    chunks = src.sqlite_handler.get_all_chunks(collection_name)
    page_ids = [chunk[1] for chunk in chunks]
    chunk_orders = [chunk[3] for chunk in chunks]
    chunks = [chunk[2] for chunk in chunks]
    create_vector_swr(collection_name, chunks, page_ids, chunk_orders, overwrite_collection=True)


if __name__ == '__main__':
    make_collection("wiki_chunks")