import sqlite3
import os
import re
import logging.config
from tqdm import tqdm
from typing import List, Tuple, Optional

from src.utils import exceptions
import config

logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class SQLiteManager:
    """
    Класс для управления SQLite базой данных: создание таблиц, сохранение страниц и чанков,
    а также обработка текста (очистка, чанкинг).
    """
    def __init__(self, db_path: str = config.SQLITE_DB_PATH,
                 page_table_name: str = config.SQLITE_PAGE_TABLE_NAME,
                 chunk_table_name: str = config.SQLITE_CHUNKS_TABLE_NAME):
        """
        Инициализация менеджера.

        Args:
            db_path (str): Путь к файлу базы данных.
            page_table_name (str): Название таблицы со страницами.
            chunk_table_name (str): Название таблицы с чанками.
        """
        logger.info("Инициализация экземпляра класса SQLiteManager")
        self.db_path = str(db_path)
        self.page_table_name = str(page_table_name)
        self.chunk_table_name = str(chunk_table_name)
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        except OSError as e:
            logger.error(f"Ошибка при создании директории для БД: {e}")
            raise exceptions.PipelineError(f"Ошибка при создании директории для БД: {e}")

    def _get_connection(self) -> sqlite3.Connection:
        """
        Внутренний метод для получения соединения.
        """
        try:
            return sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            logger.error(f"Не удалось подключиться к БД по пути {self.db_path}: {e}")
            raise exceptions.DatabaseError(f"Не удалось подключиться к БД по пути {self.db_path}: {e}")

    def create_schema(self) -> None:
        """
        Создает схему базы данных (таблицы для страниц и чанков).
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            conn.execute("PRAGMA foreign_keys = ON")

            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.page_table_name} (
                    page_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    page_url TEXT NOT NULL UNIQUE,
                    page_content TEXT NOT NULL
                )
            ''')

            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.chunk_table_name} (
                    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    page_id INTEGER NOT NULL,
                    chunk_content TEXT NOT NULL,
                    chunk_order INTEGER NOT NULL,
                    FOREIGN KEY (page_id) REFERENCES {self.page_table_name}(page_id) ON DELETE CASCADE,
                    UNIQUE (page_id, chunk_order)
                )
            ''')

            conn.commit()
            conn.close()
            logger.info(f"Схема БД инициализирована: {self.db_path} ({self.page_table_name}, {self.chunk_table_name})")
        except sqlite3.Error as e:
            logger.error(f"Ошибка инициализации схемы: {e}")
            raise exceptions.DatabaseError(f"Ошибка инициализации схемы: {e}")
        finally:
            if conn: conn.close()

    # --- Методы работы со страницами (Pages) ---

    def save_or_update_page_by_url(self, url: str, content: str) -> Optional[int]:
        """
        Сохраняет или обновляет страницу по URL. Возвращает ID страницы.
        """
        conn = None
        try:
            conn = self._get_connection()
            page_id = None
            cursor = conn.cursor()
            cursor.execute(
                f"INSERT OR REPLACE INTO {self.page_table_name} (page_url, page_content) VALUES (?, ?)",
                (url, content)
            )
            cursor.execute(f"SELECT page_id FROM {self.page_table_name} WHERE page_url = ?", (url,))
            row = cursor.fetchone()
            if row:
                page_id = row[0]
                conn.commit()
                logger.info(f"Страница '{url}' сохранена (ID: {page_id})")
            else:
                logger.error(f"Ошибка: Не удалось получить ID для {url}")

            return page_id

        except sqlite3.Error as e:
            logger.error(f"Ошибка SQLite при сохранении {url}: {e}")
            raise exceptions.DatabaseError(f"Ошибка SQLite при сохранении {url}: {e}")
        finally:
            if conn: conn.close()

    def get_all_pages(self) -> List[Tuple[int, str, str]]:
        """
        Возвращает список всех страниц: (page_id, page_url, page_content).
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT page_id, page_url, page_content FROM {self.page_table_name}")
        results = cursor.fetchall()
        conn.close()
        return results

    def get_page_by_url(self, url: str) -> Optional[Tuple[int, str, str]]:
        """
        Ищет страницу по URL.
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT page_id, page_url, page_content FROM {self.page_table_name} WHERE page_url = ?",
                (url,)
            )
            result = cursor.fetchone()
            conn.close()
            return result
        except sqlite3.Error as e:
            logger.error(f"Не удалось получить страницу по URL {url}: {e}")
            raise exceptions.DatabaseError(f"Не удалось получить страницу по URL {url}: {e}")
        finally:
            if conn: conn.close()

    # --- Методы работы с чанками (Chunks) ---

    def save_chunks(self, page_id: int, chunks: List[str]) -> None:
        """
        Сохраняет список чанков для указанной страницы, предварительно удаляя старые.
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self.chunk_table_name} WHERE page_id = ?", (page_id,))

            data_to_insert = [(page_id, chunk, i) for i, chunk in enumerate(chunks)]
            cursor.executemany(
                f"INSERT INTO {self.chunk_table_name} (page_id, chunk_content, chunk_order) VALUES (?, ?, ?)",
                data_to_insert
            )
            conn.commit()
            logger.info(f"Сохранено {len(chunks)} чанков для page_id={page_id}")
        except sqlite3.Error as e:
            logger.error(f"Ошибка сохранения чанков для ID {page_id}: {e}")
            raise exceptions.DatabaseError(f"Ошибка сохранения чанков для ID {page_id}: {e}")
        finally:
            if conn: conn.close()

    def get_all_chunks(self) -> List[Tuple[int, int, str, int]]:
        """
        Возвращает все чанки из БД, отсортированные по page_id и порядку.
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT chunk_id, page_id, chunk_content, chunk_order FROM {self.chunk_table_name} ORDER BY page_id, chunk_order"
            )
            results = cursor.fetchall()
            conn.close()
            return results
        except sqlite3.Error as e:
            logger.error(f"Ошибка при получении всех чанков: {e}")
            raise exceptions.DatabaseError(f"Ошибка при получении всех чанков: {e}")
        finally:
            if conn: conn.close()

    def get_chunks_by_page_id(self, page_id: int) -> List[Tuple[int, int, str, int]]:
        """
        Возвращает чанки для конкретной страницы.
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT chunk_id, page_id, chunk_content, chunk_order FROM {self.chunk_table_name} WHERE page_id = ? ORDER BY chunk_order",
                (page_id,)
            )
            results = cursor.fetchall()
            return results
        except sqlite3.Error as e:
            logger.error(f"Не удалось получить чанки для страницы {page_id}: {e}")
            raise exceptions.DatabaseError(f"Не удалось получить чанки для страницы {page_id}: {e}")
        finally:
            if conn: conn.close()

    # --- Методы очистки (Cleanup) ---

    def delete_page_data(self, page_id: int, delete_original_page: bool = False) -> None:
        """
        Удаляет чанки страницы. Опционально удаляет саму страницу.
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self.chunk_table_name} WHERE page_id = ?", (page_id,))
            chunks_deleted = cursor.rowcount

            if delete_original_page:
                cursor.execute(f"DELETE FROM {self.page_table_name} WHERE page_id = ?", (page_id,))
                logger.info(f"Удалена страница ID {page_id} и {chunks_deleted} чанков.")
            else:
                logger.info(f"Удалено {chunks_deleted} чанков для ID {page_id}.")

            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Ошибка при удалении данных ID {page_id}: {e}")
            raise exceptions.DatabaseError(f"Ошибка при удалении данных ID {page_id}: {e}")
        finally:
            if conn: conn.close()

    def clear_database(self, clear_pages: bool = False) -> None:
        """
        Полная очистка: удаляет все чанки. Если clear_pages=True, удаляет и страницы.
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self.chunk_table_name}")
            cursor.execute(f"UPDATE 'sqlite_sequence' SET seq = 0 WHERE name = \"{self.chunk_table_name}\"")
            if clear_pages:
                cursor.execute(f"DELETE FROM {self.page_table_name}")
                cursor.execute(f"UPDATE 'sqlite_sequence' SET seq = 0 WHERE name = \"{self.page_table_name}\"")
                logger.info("База полностью очищена (страницы и чанки).")
            else:
                logger.info("Все чанки удалены.")
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Ошибка при очистки базы данных: {e}")
            raise exceptions.DatabaseError(f"Ошибка при очистки базы данных: {e}")
        finally:
            if conn: conn.close()

    # --- Статические методы обработки текста ---

    @staticmethod
    def clean_text(content: str) -> str:
        """
        Очищает текст от ненужных данных

        Args:
            content (str): Изначальный текст.

        Returns:
            str: Изменённый текст
        """
        try:
            text = content
            text = (text.replace('\n', ' ')
                    .replace(' ↑ ', '')
                    .replace('ВНИМАНИЕ, СПОЙЛЕРЫ : Статья содержит детали сюжета игры Outer Wilds', '')
                    .replace('ВНИМАНИЕ, СПОЙЛЕРЫ : Статья содержит детали сюжета дополнения Echoes of the Eye', '')
                    .replace(' ,', ',')
                    .replace(' .', '.')
                    .replace('Заголовок: ', ''))

            link_pattern = re.compile(
                r'https?://[^\s]+|'
                r'www\.[^\s]+|'
                r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*'
            )
            text = link_pattern.sub('', text)

            square_bracket_pattern = r'\[[^\]]*\]'
            text = re.sub(square_bracket_pattern, '', text)
            return text
        except Exception as e:
            logger.error(f"Ошибка при обработке текста: {e}")
            raise exceptions.TextProcessingError(f"Ошибка при обработке текста: {e}")

    @staticmethod
    def extract_title_and_content(page: str) -> Tuple[str, str]:
        """
        Парсит сырой текст страницы на заголовок и контент.
        Ожидает формат: "Заголовок: ... \n ... Содержимое:\n ..."
        """
        try:
            title = page[:page.find('\n')]
            content = page[page.find('\n'):]
            return title, content
        except Exception as e:
            logger.error(f"Ошибка при разделении текста на заголовок и контент: {e}")
            raise exceptions.TextProcessingError(f"Ошибка при разделении текста на заголовок и контент: {e}")

    # --- Методы чанкирования ---
    @staticmethod
    def chunk_text_by_length_with_overlap(title: str, content: str, chunk_size: int = 512, overlap: int = 100) -> List[
        str]:
        """
        Разбивает текст на чанки фиксированной длины с перекрытием.
        """
        try:
            prefix = f'search_document: {title} '

            chunks = []
            start = 0
            text_length = len(content)

            while start < text_length:
                end = min(start + chunk_size, text_length)
                chunks.append(content[start:end])
                if end == text_length:
                    break
                start += chunk_size - overlap

            return [prefix + chunk for chunk in chunks]
        except Exception as e:
            logger.error(f"Ошибка при фиксированном чанкировании: {e}")
            raise exceptions.ChunkingError(f"Ошибка при фиксированном чанкировании: {e}")

    @staticmethod
    def chunk_text_by_sentences(title: str, content: str) -> List[str]:
        """
        Разбивает текст на чанки по предложениям, добавляя префикс заголовка.
        """
        try:
            parts = re.split(r'([.!?] )', content)

            sentences = [
                parts[i] + parts[i + 1]
                for i in range(0, len(parts) - 1, 2)
                if parts[i] or parts[i + 1]
            ]

            if not sentences and content:
                sentences = [content]

            prefix = f'search_document: {title} '
            chunks = [prefix + sentence for sentence in sentences]

            return chunks
        except Exception as e:
            logger.error(f"Ошибка при чанкировании по предложениям: {e}")
            raise exceptions.ChunkingError(f"Ошибка при чанкировании по предложениям: {e}")

    # --- Основной пайплайн ---

    def run_processing_pipeline(self) -> None:
        """
        Запускает процесс: чтение страниц -> парсинг -> нарезка на чанки -> сохранение чанков.
        """
        logger.info("ЗАПУСК: Пайплайн обработки SQLite.")

        self.clear_database()
        self.create_schema()

        pages = self.get_all_pages()
        if not pages:
            logger.error("Нет страниц для обработки.")
            return

        for page in tqdm(pages):
            page_id = page[0]
            raw_text = page[2]

            title, body = self.extract_title_and_content(raw_text)
            clean_body = self.clean_text(body)

            chunks = self.chunk_text_by_length_with_overlap(title, clean_body)

            self.save_chunks(page_id, chunks)

        logger.info("КОНЕЦ: Пайплайн обработки SQLite.")

if __name__ == '__main__':
    sqlite_manager = SQLiteManager()
    sqlite_manager.run_processing_pipeline()
