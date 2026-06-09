import os
from pathlib import Path

BASE_DIR = Path(__file__).parent


# Настройки данных
DATA_DIR = BASE_DIR / "data"
SQLITE_DB_PATH = DATA_DIR / "wiki_content.db"
SQLITE_PAGE_TABLE_NAME = "wiki_pages"
SQLITE_CHUNKS_TABLE_NAME = "wiki_pages_chunks_sliding"
CHROMA_DB_PATH = DATA_DIR
LOGS_DIR = BASE_DIR / "logs"
COLLECTION_NAME = "wiki_chunks_sentences"

WIKI_URL = "https://www.kubsu.ru/ru/"

# Настройки создания чанкирования
# Фиксированное с перекрытием
C_BASIC_SIZE = 512
C_BASIC_OVERLAP = 200

# Настройки Retriever-модели
R_EMBEDDING_MODEL_NAME = os.getenv("R_EMBEDDING_MODEL_NAME", "FRIDA-q8_0.gguf")
R_MAX_LENGTH = 512
R_BATCH_SIZE = 16
R_POOLING_METHOD = "cls"
R_DEVICE = os.getenv("R_DEVICE", "cpu")
R_UNLOAD_ON_GENERATION = False
R_EMBEDDING_API_URL = os.getenv("R_EMBEDDING_API_URL", "http://172.29.0.1:8083/v1") # Эмбеддинги

# Настройка NER-модели
NER_MODEL_NAME = "ru_core_news_md"

# Настройка CrossEncoder-модели
CE_LOCAL_MODEL_NAME = os.getenv("CE_LOCAL_MODEL_NAME", "bge-reranker-v2-m3-q4_k_m.gguf")
CE_DEVICE = "cpu"
CE_RERANKER_API_URL = os.getenv("CE_RERANKER_API_URL", "http://172.29.0.1:8084/v1") # Реранкер (Cross-Encoder)

# Настройки Generator-модели
G_USE_REMOTE_MODEL = os.getenv("G_USE_REMOTE_MODEL", "True").lower() == "true"
G_REMOTE_MODEL_NAME = os.getenv("G_REMOTE_MODEL_NAME", "moonshotai/kimi-k2.6:free")
G_LOCAL_MODEL_NAME = os.getenv("G_LOCAL_MODEL_NAME", "t-lite-it-1.0-q8_0.gguf")
G_DEVICE = os.getenv("G_DEVICE", "cpu")
G_LOCAL_MODEL_URL = os.getenv("G_LOCAL_MODEL_URL", "http://172.29.0.1:8082/v1")   # Генератор (LLM)

G_SYSTEM_PROMPT = """Ты полезный ассистент по вопросам связанным с университетом КубГУ. Ответь на вопрос, используя только контекст"""
G_PRINT_CHUNKS = True

# Сетапы чанкирования
# Стандартное фиксированное чанкирование с перекрытием
C_OVERLAP = {
    "C_BM25_AMOUNT": 15,
    "C_VECTOR_AMOUNT": 15,
    "C_CONCATENATE_CHUNKS_OF_PAGES": False,
    "C_USE_RERANKER": True,
    "C_RERANKER_AMOUNT": 10,
    "C_USE_KERNEL": False,
    "C_KERNEL_SIGMA": None,
    "C_KERNEL_THRESHOLD": None,
    "C_USE_DYNAMIC_EXPANSION": False,
    "C_DYNAMIC_THRESHOLD": None,
    "C_DYNAMIC_PENALTY": None
}

C_KERNEL = {
    "C_BM25_AMOUNT": 25,
    "C_VECTOR_AMOUNT": 25,
    "C_CONCATENATE_CHUNKS_OF_PAGES": True,
    "C_USE_RERANKER": False,
    "C_RERANKER_AMOUNT": None,
    "C_USE_KERNEL": True,
    "C_KERNEL_SIGMA": 4.0,
    "C_KERNEL_THRESHOLD": 0.08,
    "C_USE_DYNAMIC_EXPANSION": False,
    "C_DYNAMIC_THRESHOLD": None,
    "C_DYNAMIC_PENALTY": None
}

C_DYNAMIC = {
    "C_BM25_AMOUNT": 10,
    "C_VECTOR_AMOUNT": 10,
    "C_CONCATENATE_CHUNKS_OF_PAGES": True,
    "C_USE_RERANKER": False,
    "C_RERANKER_AMOUNT": None,
    "C_USE_KERNEL": False,
    "C_KERNEL_SIGMA": None,
    "C_KERNEL_THRESHOLD": None,
    "C_USE_DYNAMIC_EXPANSION": True,
    "C_DYNAMIC_THRESHOLD": 0.5,
    "C_DYNAMIC_PENALTY": 0.05
}

TEST_LOCAL_C_OVERLAP = {
    "C_BM25_AMOUNT": 50,
    "C_VECTOR_AMOUNT": 50,
    "C_CONCATENATE_CHUNKS_OF_PAGES": False,
    "C_USE_RERANKER": True,
    "C_RERANKER_AMOUNT": 10,
    "C_USE_KERNEL": False,
    "C_KERNEL_SIGMA": None,
    "C_KERNEL_THRESHOLD": None,
    "C_USE_DYNAMIC_EXPANSION": False,
    "C_DYNAMIC_THRESHOLD": None,
    "C_DYNAMIC_PENALTY": None
}

TEST_LOCAL_C_KERNEL = {
    "C_BM25_AMOUNT": 3,
    "C_VECTOR_AMOUNT": 3,
    "C_CONCATENATE_CHUNKS_OF_PAGES": True,
    "C_USE_RERANKER": False,
    "C_RERANKER_AMOUNT": None,
    "C_USE_KERNEL": True,
    "C_KERNEL_SIGMA": 4.0,
    "C_KERNEL_THRESHOLD": 0.13,
    "C_USE_DYNAMIC_EXPANSION": False,
    "C_DYNAMIC_THRESHOLD": None,
    "C_DYNAMIC_PENALTY": None
}

C_LOW_AMOUNT_KERNEL = {
    "C_BM25_AMOUNT": 10,
    "C_VECTOR_AMOUNT": 10,
    "C_CONCATENATE_CHUNKS_OF_PAGES": True,
    "C_USE_RERANKER": False,
    "C_RERANKER_AMOUNT": None,
    "C_USE_KERNEL": True,
    "C_KERNEL_SIGMA": 4.0,
    "C_KERNEL_THRESHOLD": 0.13,
    "C_USE_DYNAMIC_EXPANSION": False,
    "C_DYNAMIC_THRESHOLD": None,
    "C_DYNAMIC_PENALTY": None
}

C_LOW_AMOUNT_OVERLAP = {
    "C_BM25_AMOUNT": 30,
    "C_VECTOR_AMOUNT": 30,
    "C_CONCATENATE_CHUNKS_OF_PAGES": False,
    "C_USE_RERANKER": True,
    "C_RERANKER_AMOUNT": 5,
    "C_USE_KERNEL": False,
    "C_KERNEL_SIGMA": None,
    "C_KERNEL_THRESHOLD": None,
    "C_USE_DYNAMIC_EXPANSION": False,
    "C_DYNAMIC_THRESHOLD": None,
    "C_DYNAMIC_PENALTY": None
}

SELECTED_C_CONFIG = C_KERNEL

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)