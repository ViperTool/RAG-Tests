import config

import os
import re
import sqlite3
from typing import List, Tuple, Optional


def create_content(page_table_name: str = str(config.SQLITE_PAGE_TABLE_NAME),
                   chunk_table_name: str = str(config.SQLITE_CHUNKS_TABLE_NAME),
                   db_path: str = str(config.SQLITE_DB_PATH)):
    """
    Метод для работы с таблицей со страницами и чанками.
    Создает SQLite базу данных и таблицу для хранения содержимого страниц.
    Также создает таблицу для хранения чанков, связанную с основной таблицей.

    Args:
        page_table_name (str): Название таблицы со страницами.
        chunk_table_name (str): Название таблицы с чанками.
        db_path (str): Путь к файлу базы данных.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Таблица для чанков
    conn.execute("PRAGMA foreign_keys = ON")

    # Создание таблицы pages
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {page_table_name} (
            page_id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_url TEXT NOT NULL,
            page_content TEXT NOT NULL
        )
    ''')

    # Создание таблицы sentence_chunks
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {chunk_table_name} (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id INTEGER NOT NULL,
            chunk_content TEXT NOT NULL,
            chunk_order INTEGER NOT NULL,
            FOREIGN KEY (page_id) REFERENCES pages(page_id) ON DELETE CASCADE,
            UNIQUE (page_id, chunk_order)
        )
    ''')

    conn.commit()
    conn.close()
    print(f"База данных содержимого и чанков создана: {db_path}, наименование таблицы со страницами: {page_table_name}, наименование таблицы с чанками: {chunk_table_name}")


def save_or_update_page_by_url(url: str, content: str, table_name: str = str(config.SQLITE_PAGE_TABLE_NAME), db_path: str = str(config.SQLITE_DB_PATH)) -> Optional[int]:
    """
    Метод для работы с таблицей со страницами.
    Сохраняет (или обновляет, если URL уже существует) содержимое одной страницы в базу данных по URL.
    Возвращает ID сохраненной/обновленной страницы.

    Args:
        url (str): URL страницы.
        content (str): Содержимое страницы.
        table_name (str): Название таблицы
        db_path (str): Путь к файлу базы данных.

    Returns:
        Optional[int]: ID сохраненной/обновленной страницы или None в случае ошибки.
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    page_id = None
    try:
        cursor.execute(
            f"INSERT OR REPLACE INTO {table_name} (page_url, page_content) VALUES (?, ?)",
            (url, content)
        )
        # Получаем ID вставленной или замененной строки
        cursor.execute(f"SELECT page_id FROM {table_name} WHERE page_url = ?", (url,))
        row = cursor.fetchone()
        if row:
            page_id = row[0]
            conn.commit()
            print(f"Содержимое страницы сохранено/обновлено в БД: {url}, ID: {page_id}")
        else:
            # Теоретически маловероятно при INSERT OR REPLACE, но на всякий случай
            print(f"Ошибка: Не удалось получить ID после INSERT OR REPLACE для {url}.")
            page_id = None
    except sqlite3.Error as e:
        print(f"1 Ошибка SQLite при сохранении/обновлении {url}: {e}")
        page_id = None # Возвращаем None в случае ошибки
    finally:
        conn.close()
    return page_id # Возвращаем ID страницы


def save_or_update_page_by_id(page_id: int, content: str, table_name: str = str(config.SQLITE_PAGE_TABLE_NAME), db_path: str = str(config.SQLITE_DB_PATH)) -> Optional[bool]:
    """
    Метод для работы с таблицей со страницами.
    Обновляет содержимое одной страницы в базу данных по её ID.
    Возвращает True, если обновление прошло успешно, False, если запись не найдена или произошла ошибка.

    Args:
        page_id (int): ID страницы.
        content (str): Новое содержимое страницы.
        table_name (str): Название таблицы.
        db_path (str): Путь к файлу базы данных.

    Returns:
        Optional[bool]: True если обновлено, False если не найдено или ошибка.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    success = False
    try:
        cursor.execute(
            f"UPDATE {table_name} SET page_content = ? WHERE page_id = ?",
            (content, page_id)
        )
        # Проверяем, была ли затронута хотя бы одна строка
        if cursor.rowcount > 0:
            conn.commit()
            print(f"Содержимое страницы с ID {page_id} обновлено.")
            success = True
        else:
            print(f"Предупреждение: Страница с ID {page_id} не найдена. Нечего обновлять.")
            success = False # Возвращаем False, если запись не найдена
    except sqlite3.Error as e:
        print(f"2 Ошибка SQLite при обновлении содержимого для ID {page_id}: {e}")
        success = False # Возвращаем False в случае ошибки
    finally:
        conn.close()
    return success


def save_chunks(page_id: int, chunks: List[str], table_name: str = str(config.SQLITE_CHUNKS_TABLE_NAME), db_path: str = str(config.SQLITE_DB_PATH)):
    """
    Метод для работы с таблицей с чанками.
    Сохраняет список чанков в базу данных, связывая их с ID страницы.

    Args:
        page_id (int): ID страницы из таблицы config.SQLITE_PAGE_TABLE_NAME.
        chunks (List[str]): Список строк, представляющих чанки текста.
        table_name (str): Название таблицы
        db_path (str): Путь к файлу базы данных.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        # Удаляем существующие чанки для этой страницы (если нужно обновить)
        cursor.execute(f"DELETE FROM {table_name} WHERE page_id = ?", (page_id,))

        # Вставляем новые чанки
        for i, chunk_content in enumerate(chunks):
            cursor.execute(
                f"INSERT INTO {table_name} (page_id, chunk_content, chunk_order) VALUES (?, ?, ?)",
                (page_id, chunk_content, i) # Сохраняем порядок
            )
        conn.commit()
        print(f"Сохранено {len(chunks)} чанков для страницы ID {page_id}")
    except sqlite3.Error as e:
        print(f"3 Ошибка SQLite при сохранении чанков для ID {page_id}: {e}")
    finally:
        conn.close()


def get_all_pages(table_name: str = str(config.SQLITE_PAGE_TABLE_NAME), db_path: str = str(config.SQLITE_DB_PATH)) -> List[Tuple[int, str, str]]:
    """
    Метод для работы с таблицей со страницами.
    Извлекает все содержимое из базы данных.

    Args:
        table_name (str): Название таблицы.
        db_path (str): Путь к файлу базы данных.

    Returns:
        List[Tuple[int, str, str]]: Список кортежей (page_id, page_url, page_content).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT page_id, page_url, content FROM {table_name}")
    results = cursor.fetchall()
    conn.close()
    return results


def get_all_chunks(table_name: str = str(config.SQLITE_PAGE_TABLE_NAME), db_path: str = str(config.SQLITE_DB_PATH)) -> List[Tuple[int, int, str, int]]:
    """
    Метод для работы с таблицей с чанками.
    Извлекает все чанки из базы данных.

    Args:
        table_name (str): Название таблицы
        db_path (str): Путь к файлу базы данных.

    Returns:
        List[Tuple[int, int, str, int]]: Список всех кортежей (chunk_id, page_id, chunk_content, chunk_order).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT chunk_id, page_id, chunk_content, chunk_order FROM {table_name} ORDER BY page_id, chunk_order")
    results = cursor.fetchall()
    conn.close()
    return results


def get_page_by_url(url: str, table_name: str = str(config.SQLITE_PAGE_TABLE_NAME), db_path: str = str(config.SQLITE_DB_PATH)) -> Tuple[int, str, str] | None:
    """
    Метод для работы с таблицей со страницами.
    Извлекает содержимое страницы по URL из базы данных.

    Args:
        url (str): URL страницы.
        table_name (str): Название таблицы.
        db_path (str): Путь к файлу базы данных.

    Returns:
        Tuple[int, str, str] | None: Кортеж (id, url, content) или None, если не найдено.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT page_id, page_url, page_content FROM {table_name} WHERE page_url = ?", (url,))
    result = cursor.fetchone()
    conn.close()
    return result


def get_page_by_id(page_id: int, table_name: str = str(config.SQLITE_PAGE_TABLE_NAME), db_path: str = str(config.SQLITE_DB_PATH)) -> Tuple[int, str, str] | None:
    """
    Метод для работы с таблицей со страницами.
    Извлекает содержимое страницы по ID из базы данных.

    Args:
        page_id (int): ID страницы.
        table_name (str): Название таблицы.
        db_path (str): Путь к файлу базы данных.

    Returns:
        Tuple[int, str, str] | None: Кортеж (id, url, content) или None, если не найдено.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, url, content FROM {table_name} WHERE id = ?", (page_id,))
    result = cursor.fetchone()
    conn.close()
    return result


def get_chunks_by_page_id(page_id: int, table_name: str = str(config.SQLITE_CHUNKS_TABLE_NAME), db_path: str = str(config.SQLITE_DB_PATH)) -> List[Tuple[int, int, str, int]]:
    """
    Метод для работы с таблицей с чанками.
    Извлекает все чанки для заданного ID страницы.

    Args:
        page_id (int): ID страницы из таблицы wiki_pages.
        table_name (str): Название таблицы.
        db_path (str): Путь к файлу базы данных.

    Returns:
        List[Tuple[int, int, str, int]]: Список кортежей (chunk_id, page_id, chunk_content, chunk_order).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT chunk_id, page_id, chunk_content, chunk_order FROM {table_name} WHERE page_id = ? ORDER BY chunk_order", (page_id,))
    results = cursor.fetchall()
    conn.close()
    return results


def delete_page_and_chunks(page_id: int, table_name: str = str(config.SQLITE_CHUNKS_TABLE_NAME), delete_original: bool = False, db_path: str = str(config.SQLITE_DB_PATH)):
    """
    Метод для работы с таблицей с чанками И (по желанию) со страницами.
    Удаляет чанки, связанные с указанным ID страницы.
    При необходимости также удаляет оригинальный документ из wiki_pages.

    Args:
        page_id (int): ID страницы, чанки которой нужно удалить.
        table_name (str): Название таблицы.
        delete_original (bool): Если True, удаляет также запись из wiki_pages.
                                Если False, удаляет только чанки.
        db_path (str): Путь к файлу базы данных.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(f"DELETE FROM {table_name} WHERE page_id = ?", (page_id,))
        deleted_chunks_count = cursor.rowcount
        print(f"Удалено {deleted_chunks_count} чанков для страницы ID {page_id}.")

        if delete_original:
            cursor.execute(f"DELETE FROM {table_name} WHERE page_id = ?", (page_id,))
            deleted_pages_count = cursor.rowcount
            if deleted_pages_count > 0:
                print(f"Оригинальный документ (ID {page_id}) также удален.")
            else:
                print(f"Предупреждение: Оригинальный документ с ID {page_id} не найден для удаления.")
        else:
            print(f"Оригинальный документ (ID {page_id}) оставлен в базе данных.")

        conn.commit()
    except sqlite3.Error as e:
        print(f"4 Ошибка SQLite при удалении для ID {page_id}: {e}")
    finally:
        conn.close()


def clear_db(table_name: str = str(config.SQLITE_CHUNKS_TABLE_NAME), db_path: str = str(config.SQLITE_DB_PATH), clear_pages: bool = False):
    """
    Метод для работы с таблицей с чанками И (по желанию) со страницами.
    Очищает базу данных с чанками.

    Args:
        table_name (str): Название таблицы.
        db_path (str): Путь до базы данных,
        clear_pages (bool): Очистить ли базу данных со страницами.
    """

    pages = get_all_pages(db_path)
    for page in pages:
        delete_page_and_chunks(page_id=page[0], table_name=table_name, delete_original=clear_pages, db_path=db_path)


def form_chunks(title: str, content: str) -> List[str]:
    """
    Формирует чанки. Разделение осуществляется по точкам и ограничению длины полученных предложений.

    Args:
        title (str): Оглавление страницы,
        content (str): Содержимое страницы.

    Returns:
        List[str]: Список полученных предложений.
    """
    total_chunks = []
    current_chunk = 'search_document: ' + title + ' '
    length_of_current_chunk = 0
    splitted_content = content.split('.')
    for chunk in splitted_content:
        if length_of_current_chunk + len(chunk) < 768:
            current_chunk = current_chunk + chunk
            length_of_current_chunk += len(chunk)
        else:
            total_chunks.append(current_chunk)
            current_chunk = 'search_document: ' + title + ' '
            length_of_current_chunk = 0

    return total_chunks if len(total_chunks) > 0 else [current_chunk]


def form_chunks_swr(title: str, content: str) -> List[str]:
    """
    Формирует чанки. Разделение осуществляется по предложениям.

    Args:
        title (str): Оглавление страницы,
        content (str): Содержимое страницы.

    Returns:
        List[str]: Список полученных предложений.
    """
    parts = re.split(r'([.!?] )', content)

    sentences = [
        parts[i] + parts[i + 1]
        for i in range(0, len(parts) - 1, 2)
        if parts[i] or parts[i + 1]
    ]

    prefix = f'search_document: {title} '
    total_chunks = [prefix + sentence for sentence in sentences]

    return total_chunks if total_chunks else [prefix.strip()]


def remove_alpha_pages(db_path: str = str(config.SQLITE_DB_PATH)):
    """
    Удаляет все страницы, в названии которых фигурирует (Альфа)
    """
    pages = get_all_pages(db_path)
    for page in pages:
        text = page[2]
        text = text[:text.find('\n')]
        if '(Альфа)' in text:
            delete_page_and_chunks(page[0], delete_original=True)


def clean_text(content: str) -> str:
    """
    Очищает текст от ненужных данных

    Args:
        content (str): Изначальный текст.

    Returns:
        str: Изменённый текст
    """

    text = content
    text = (text.replace('\n', ' ')
            .replace(' ↑ ', '')
            .replace('ВНИМАНИЕ, СПОЙЛЕРЫ : Статья содержит детали сюжета игры Outer Wilds', '')
            .replace('ВНИМАНИЕ, СПОЙЛЕРЫ : Статья содержит детали сюжета дополнения Echoes of the Eye', '')
            .replace(' ,', ',')
            .replace(' .', '.')
            .replace('Заголовок: ', ''))
    if 'Содержание 1' in text:
        desc = text[text.find('Содержание 1'):text.find('[ ]')]
        text = text.replace(desc, '')

    link_pattern = re.compile(
        r'https?://[^\s]+|'
        r'www\.[^\s]+|'
        r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*'
    )
    text = link_pattern.sub('', text)

    square_bracket_pattern = r'\[[^\]]*\]'
    text = re.sub(square_bracket_pattern, '', text)
    return text


def get_title_and_content_of_page(page: str) -> Tuple[str, str]:
    """
        Получает заголовок и содержимое страницы

        Args:
            page (str): Страница в сыром виде.

        Returns:
            Tuple[str, str]: Заголовок и содержимое страницы соответственно.
        """
    title = page[page.find("Заголовок: ")+len("Заголовок: "):page.find('\n')]
    content = page[page.find("Содержимое:\n")+len("Содержимое:\n"):]
    return title, content


def run_sqlite_pipeline(page_table_name: str = str(config.SQLITE_PAGE_TABLE_NAME), chunk_table_name: str = str(config.SQLITE_CHUNKS_TABLE_NAME), db_path: str = str(config.SQLITE_DB_PATH)):
    """
    Основная функция, которая запускает создание таблицы чанков.

    Args:
        page_table_name (str) Название таблицы со страницами.
        chunk_table_name (str): Название таблицы с чанками.
        db_path (str): Путь до базы данных,
    """

    create_content(page_table_name=page_table_name, chunk_table_name=chunk_table_name)
    pages = get_all_pages(table_name="wiki_pages", db_path=db_path)
    for page in pages:
        print(page)
        title, content = get_title_and_content_of_page(page[2])
        chunks = form_chunks_swr(title, content)
        save_chunks(page_id=page[0], chunks=chunks, table_name=chunk_table_name, db_path=db_path)
