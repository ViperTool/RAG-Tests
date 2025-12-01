import os
from pathlib import Path

# Определяем корень проекта
BASE_DIR = Path(__file__).parent

# Настройки данных
DATA_DIR = BASE_DIR / "data"
SQLITE_DB_PATH = DATA_DIR / "wiki_content.db"
SQLITE_PAGE_TABLE_NAME = "wiki_pages"
SQLITE_CHUNKS_TABLE_NAME = "wiki_pages_chunks"
CHROMA_DB_PATH = DATA_DIR
COLLECTION_NAME = "wiki_chunks"

WIKI_URL = "https://outer-wilds.fandom.com/ru/wiki/"

# Настройки Retriever-модели
R_EMBEDDING_MODEL_NAME = "ai-forever/FRIDA"
R_MAX_LENGTH = 512
R_BATCH_SIZE = 16
R_POOLING_METHOD = "cls"
R_DEVICE = "cpu"
R_UNLOAD_ON_GENERATION = False

# Настройки Generator-модели
G_MODEL_NAME = "Qwen/Qwen3-1.7B"
# G_SYSTEM_PROMPT = "Роль: Ты — Бортовой Компьютер корабля исследователя Outer Wilds.\nЗадача: Проанализировать загруженные логи (контекст) и вывести справку по запросу пилота.\n\nИнструкции:\n- Анализируй только предоставленные данные. Не выдумывай факты, которых нет в логах.\n- Стиль общения: лаконичный, сильно роботизированный и полезный (как интерфейс корабля).\n- Если данные повреждены или отсутствуют (нет в контексте), сообщи: 'Запись в бортовом журнале не найдена'.\n- Помечай важные предупреждения (например, об опасных локациях), если они есть в тексте."
G_SYSTEM_PROMPT = "Используй приведенный ниже контекст из вики Outer Wilds, чтобы ответить на вопрос. Если ответа нет в контексте, скажи \"Я не знаю\". Ни в коем случае не придумывай информацию."
G_PRINT_CHUNKS = True

os.makedirs(DATA_DIR, exist_ok=True)
