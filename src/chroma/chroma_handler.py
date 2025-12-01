import config

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import List
from tqdm import tqdm

from src.core.embeddings import T5EmbeddingService
from src.sqlite.sqlite_handler import get_all_chunks


class ChromaEmbeddingAdapter(EmbeddingFunction):
    """
    Адаптер, который позволяет ChromaDB использовать T5EmbeddingService.
    ChromaDB ожидает класс с методом __call__, принимающим список текстов.
    """
    def __init__(self):
        self.service = T5EmbeddingService()
        self.service.load()

    def __call__(self, input: Documents) -> Embeddings:
        embeddings_np = self.service.encode(input)
        return embeddings_np.tolist()


def get_chroma_client():
    """Единая точка доступа к клиенту, чтобы не путать пути."""
    return chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))


def create_vector(chunks: List[str], page_ids: List[int], chunk_orders: List[int], collection_name: str = str(config.COLLECTION_NAME), overwrite_collection: bool = False):
    """
    Создает коллекцию в ChromaDB из списка чанков.

    Args:
        chunks (List[str]): Массив чанков.
        page_ids (List[int]): Массив ID страниц.
        chunk_orders (List[int]): Массив ID чанков.
        collection_name (str): Наименование коллекции.
        overwrite_collection (bool): Перезаписывать ли коллекцию.
    """
    client = get_chroma_client()

    embedding_function = ChromaEmbeddingAdapter()

    if overwrite_collection:
        try:
            client.delete_collection(name=collection_name)
            print(f"Коллекция '{collection_name}' удалена.")
        except chromadb.errors.NotFoundError:
            pass

    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

    metadatas = [
        {"page_id": str(p_id), "chunk_order": str(c_ord)}
        for p_id, c_ord in zip(page_ids, chunk_orders)
    ]

    ids = [f"chunk_{p_id}_{c_ord}" for p_id, c_ord in zip(page_ids, chunk_orders)]

    batch_limit = 512
    total = len(chunks)

    print(f"Начинаем добавление {total} элементов в Chroma...")
    for i in tqdm(range(0, total, batch_limit)):
        end = min(i + batch_limit, total)

        collection.add(
            documents=chunks[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
        print(f"Обработано {end}/{total} чанков")

    print(f"Коллекция '{collection_name}' успешно создана.")


def load_collection(collection_name: str = str(config.COLLECTION_NAME)):
    """
    Загружает коллекцию для использования в поиске (RAG).
    Возвращает: corpus (тексты), metadatas, doc_ids

    Args:
        collection_name (str): Наименование коллекции.
    """
    client = get_chroma_client()

    collection = client.get_collection(
        name=collection_name,
        embedding_function=ChromaEmbeddingAdapter()
    )

    data = collection.get(include=["documents", "metadatas", "embeddings"])

    return data["documents"], data["metadatas"], data["ids"]


def delete_collection(collection_name: str):
    """Удаляет коллекцию по имени."""
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
        print(f"Коллекция {collection_name} удалена.")
    except ValueError:
        print(f"Коллекция {collection_name} не найдена.")


def run_chroma_pipeline(collection_name: str = str(config.COLLECTION_NAME)):
    """
    Основная функция, которая читает чанки из SQLite3 и сохраняет их в ВБД Chroma.

    Args:
        collection_name (str): Наименование коллекции.
    """
    print("Чтение чанков из SQLite...")
    raw_chunks = get_all_chunks(config.SQLITE_CHUNKS_TABLE_NAME)

    if not raw_chunks:
        print("SQLite база пуста или чанки не найдены.")
        return

    page_ids = [row[1] for row in raw_chunks]
    chunks_text = [row[2] for row in raw_chunks]
    chunk_orders = [row[3] for row in raw_chunks]

    create_vector(
        chunks=chunks_text,
        page_ids=page_ids,
        chunk_orders=chunk_orders,
        collection_name=collection_name,
        overwrite_collection=True
    )
