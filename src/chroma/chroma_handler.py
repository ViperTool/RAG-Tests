import chromadb
import logging.config
from chromadb import Documents, EmbeddingFunction, Embeddings, errors
from typing import List, Tuple, Optional
from tqdm import tqdm

from src.utils import exceptions
import config
from src.core.embeddings import EmbeddingService
from src.sqlite.sqlite_handler import SQLiteManager

logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class ChromaEmbeddingAdapter(EmbeddingFunction):
    """
    Адаптер, который позволяет ChromaDB использовать EmbeddingService.
    ChromaDB ожидает класс с методом __call__, принимающим список текстов.
    """
    def __init__(self):

        try:
            self.service = EmbeddingService()
            self.service.load()
        except Exception as e:
            logger.error(f"Ошибка инициализации эмбеддинг-адаптера: {e}")
            raise exceptions.EmbeddingGenerationError(f"Ошибка инициализации эмбеддинг-адаптера: {e}")

    def __call__(self, input: Documents) -> Embeddings:
        try:
            embeddings_np = self.service.encode(input)
            return embeddings_np.tolist()
        except Exception as e:
            logger.error(f"Ошибка при генерации эмбеддинга в адаптере: {e}")
            raise exceptions.EmbeddingGenerationError(f"Ошибка при генерации эмбеддинга в адаптере: {e}")


class ChromaManager:
    """
    Класс для управления векторной базой данных ChromaDB.
    Отвечает за подключение, создание коллекций, добавление векторов и поиск.
    """

    def __init__(self, db_path: str = config.CHROMA_DB_PATH, default_collection_name: str = config.COLLECTION_NAME):
        """
        Инициализация менеджера.

        Args:
            db_path (str): Путь к файлу/директории ChromaDB.
            default_collection_name (str): Имя коллекции по умолчанию.
        """
        logger.info("Инициализация экземпляра класса ChromaManager")
        self.db_path = str(db_path)
        self.default_collection_name = default_collection_name
        try:
            self.embedding_function = ChromaEmbeddingAdapter()
            self.sqlite_manager = SQLiteManager()
            self.client = chromadb.PersistentClient(path=self.db_path)
        except Exception as e:
            logger.error(f"Ошибка инициализации менеджера векторной базы: {e}")
            raise exceptions.VectorDBError(f"Ошибка инициализации менеджера векторной базы: {e}")

    # --- Основные операции с коллекциями ---

    def create_vector_collection(self,
                                 chunks: List[str],
                                 page_ids: List[int],
                                 chunk_orders: List[int],
                                 collection_name: Optional[str] = None,
                                 overwrite: bool = False) -> None:
        """
        Создает и заполняет коллекцию в ChromaDB из списков данных.

        Args:
            chunks (List[str]): Список текстов (чанков).
            page_ids (List[int]): Список ID страниц.
            chunk_orders (List[int]): Список порядковых номеров чанков.
            collection_name (str, optional): Имя коллекции. Если None, берется из self.default_collection_name.
            overwrite (bool): Если True, удаляет существующую коллекцию перед созданием.
        """
        target_name = collection_name or self.default_collection_name

        try:
            if overwrite:
                self.delete_collection(target_name)

            collection = self.client.get_or_create_collection(
                name=target_name,
                embedding_function=self.embedding_function
            )

            metadatas = [
                {"page_id": str(p_id), "chunk_order": str(c_ord)}
                for p_id, c_ord in zip(page_ids, chunk_orders)
            ]

            ids = [f"chunk_{p_id}_{c_ord}" for p_id, c_ord in zip(page_ids, chunk_orders)]

            batch_limit = 512
            total = len(chunks)

            logger.info(f"Начинаем добавление {total} элементов в коллекцию '{target_name}'...")

            for i in tqdm(range(0, total, batch_limit), desc="Indexing chunks"):
                end = min(i + batch_limit, total)

                collection.add(
                    documents=chunks[i:end],
                    metadatas=metadatas[i:end],
                    ids=ids[i:end]
                )

            logger.info(f"Коллекция '{target_name}' успешно обновлена.")
        except Exception as e:
            logger.error(f"Ошибка при создании векторной коллекции: {e}")
            raise exceptions.VectorDBError(f"Ошибка при создании векторной коллекции: {e}")


    def load_collection_data(self, collection_name: Optional[str] = None) -> Tuple[List[str], List[dict], List[str]]:
        """
        Загружает данные коллекции (документы, метаданные, ID).
        """
        target_name = collection_name or self.default_collection_name

        try:
            collection = self.client.get_collection(
                name=target_name,
                embedding_function=self.embedding_function
            )

            data = collection.get(include=["documents", "metadatas", "embeddings"])
            return data["documents"], data["metadatas"], data["ids"]

        except Exception as e:
            logger.error(f"Ошибка при загрузке коллекции: {e}")
            raise exceptions.VectorDBError(f"Ошибка при загрузке коллекции: {e}")

    def delete_collection(self, collection_name: str) -> None:
        """
        Удаляет коллекцию по имени.
        """
        a = str(input(f"Вы уверены, что хотите удалить коллекцию {collection_name}? Для подтверждения напишите Y или y\n"))
        if a.lower() == "y":
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"Коллекция '{collection_name}' удалена.")
            except (ValueError, chromadb.errors.NotFoundError):
                logger.warning(f"Коллекция '{collection_name}' не найдена или уже удалена.")

    # --- Основной пайплайн ---

    def run_chroma_pipeline(self,
                            sqlite_table_name: Optional[str] = None,
                            collection_name: Optional[str] = None) -> None:
        """
        Читает чанки из SQLite (через callback-функцию) и сохраняет их в Chroma.

        Args:
            sqlite_table_name (str, optional): Имя таблицы в SQLite.
            collection_name (str, optional): Имя целевой коллекции.
        """
        logger.info("ЗАПУСК: Пайплайн Chroma")

        sqlite_table_name = sqlite_table_name or config.SQLITE_CHUNKS_TABLE_NAME
        target_name = collection_name or self.default_collection_name

        logger.info(f"Чтение чанков из таблицы '{sqlite_table_name}'...")

        raw_chunks = self.sqlite_manager.get_all_chunks()

        if not raw_chunks:
            logger.info("База данных пуста или чанки не найдены.")
            return

        page_ids = [row[1] for row in raw_chunks]
        chunks_text = [row[2] for row in raw_chunks]
        chunk_orders = [row[3] for row in raw_chunks]

        self.create_vector_collection(
            chunks=chunks_text,
            page_ids=page_ids,
            chunk_orders=chunk_orders,
            collection_name=target_name,
            overwrite=True
        )

        logger.info("КОНЕЦ: Пайплайн Chroma")

if __name__ == '__main__':
    chroma_manager = ChromaManager()
    chroma_manager.run_chroma_pipeline()