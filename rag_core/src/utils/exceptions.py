class PipelineError(Exception):
    """Базовый класс для всех ошибок пайплайна."""
    pass

# --- Группа ошибок базы данных ---
class DatabaseError(PipelineError):
    """Ошибки, связанные с БД (SQLite, Chroma)."""
    pass

class TextProcessingError(DatabaseError):
    """Ошибки при очистке или нарезке текста."""
    pass

class LLMGenerationError(DatabaseError):
    """Ошибки при обращении к API LLM."""
    pass

class ChunkingError(DatabaseError):
    """Ошибки при чанкировании"""
    pass

# --- Группа ошибок векторной базы ---
class VectorDBError(PipelineError):
    """Базовая ошибка для операций с векторной БД."""
    pass

class CollectionNotFoundError(VectorDBError):
    """Попытка обратиться к несуществующей коллекции."""
    pass

class EmbeddingGenerationError(VectorDBError):
    """Ошибка при генерации эмбеддингов."""
    pass

class IndexingError(VectorDBError):
    """Ошибка при добавлении данных в индекс (напр. дубликаты ID или битые вектора)."""
    pass

# --- Группа ошибок эмбеддингов ---
class EmbeddingError(PipelineError):
    """Базовая ошибка для операций с эмбеддингами."""
    pass

class EmbeddingModelLoadingError(EmbeddingError):
    """Ошибка загрузки encoder-модели."""
    pass

class EncodeError(EmbeddingError):
    """Ошибка при генерации эмбеддингов"""
    pass

# --- Группа обработок парсера ---
class NetworkError(PipelineError):
    """Ошибки сети (таймауты, 404, 429, отсутствие интернета)."""
    pass

class ParsingError(PipelineError):
    """Ошибка при разборе HTML (BeautifulSoup)."""
    pass

# --- Группа ошибок генератора ---
class LLMError(PipelineError):
    """Базовый класс для ошибок генерации текста (LLM)."""
    pass

class ModelLoadingError(LLMError):
    """Ошибка загрузки весов модели (нет файла, битый конфиг, нет места в VRAM)."""
    pass

class GenerationError(LLMError):
    """Ошибка во время inference (OOM во время генерации, некорректные инпуты)."""
    pass

class APIError(LLMError):
    """Ошибка при работе с внешним API (OpenRouter)."""
    pass

# --- Группа ошибок RAG-движка ---
class RAGError(PipelineError):
    """Базовый класс для ошибок RAG-движка"""
    pass

class InitializationError(RAGError):
    """Ошибка во время инициализации RAG-движка"""
    pass