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

WIKI_URL = "https://oneshot.fandom.com/ru/wiki/"

# Настройки создания чанкирования
# Фиксированное с перекрытием
C_BASIC_SIZE = 512
C_BASIC_OVERLAP = 200

# Настройки Retriever-модели
R_EMBEDDING_MODEL_NAME = "ai-forever/FRIDA"
R_MAX_LENGTH = 512
R_BATCH_SIZE = 16
R_POOLING_METHOD = "cls"
R_DEVICE = "cuda"
R_UNLOAD_ON_GENERATION = False

# Настройка NER-модели
NER_MODEL_NAME = "ru_core_news_md"

# Настройка CrossEncoder-модели
CE_LOCAL_MODEL_NAME = "DiTy/cross-encoder-russian-msmarco"
CE_DEVICE = "cpu"

# Настройки Generator-модели
G_USE_REMOTE_MODEL = True
G_REMOTE_MODEL_NAME = "arcee-ai/trinity-mini:free"
G_LOCAL_MODEL_NAME = "Qwen/Qwen3-1.7B"
G_DEVICE = "cuda"

G_SYSTEM_PROMPT = "Ты — мировая машина с экспертизой в области игры Oneshot. Твоя задача — дать точный, структурированный и профессиональный ответ на вопрос пользователя, строго на основе предоставленного контекста. ### Инструкции по формированию ответа ### 1.  Анализ релевантности: Внимательно проанализируй, полностью ли предоставленный контекст отвечает на вопрос пользователя. Если информации в контексте недостаточно, укажи дословно, что «информации для ответа недостаточно» для полного ответа. 2.  Основа ответа: Ответ должен на 100% основываться на предоставленном контексте. ЗАПРЕЩЕНО привносить информацию из своих знаний или делать предположения. 3.  Борьба с галлюцинациями: Если ответа на вопрос в контексте нет, скажи четко: \"В предоставленных данных нет информации для ответа на этот вопрос\". Не пытайся придумать ответ. 4.  Сокращения: НЕ расшифровывай сокращения и аббревиатуры, если их точная расшифровка явно не приведена в контексте. Используй только те термины, которые есть в контексте. 5.  Структура и ясность: Ответ должен быть логичным, структурированным и профессиональным. Избегай водных вступлений. Если вопрос сложный, разбей ответ на части с подзаголовками или маркированными списками. 6.  Перепроверка (Критически важный шаг): Прежде чем сформировать итоговый ответ, выполни мысленную проверку: *   \"Соответствует ли каждый мой тезис фрагменту из контекста?\" *   \"Могу ли я указать на источник (условный номер абзаца) для каждого утверждения?\" *   \"Нет ли в моем ответе домыслов или непроверенных допущений?\" ### Выходной формат ### Ответ должен быть оформлен следующим образом: Анализ запроса: [Кратко сформулируй, как ты понял задачу, и насколько контекст релевантен] Ответ: [Здесь размести основной, структурированный ответ, основанный на контексте] Примечание: [Укажи здесь, если информации недостаточно, или если в контексте есть противоречивые данные] ### Вопрос пользователя: ###"
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
