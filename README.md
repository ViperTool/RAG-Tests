# RAG Assistant для КубГУ

Интеллектуальный ассистент по базе знаний университета, построенный на архитектуре RAG (Retrieval Augmented Generation).

Проект реализует **гибридный поиск** (BM25 + Dense Embeddings) с кастомными алгоритмами расширения контекста: **Gaussian Kernel** (восстановление связности текста через гауссово сглаживание) и **Dynamic Expansion** (семантическое расширение по соседним чанкам). Поддерживается реранкинг через Cross-Encoder и автоматизированная оценка качества через **RAGAS**.

### Особенности
*   **Стек:** Python 3.12, FastAPI, ChromaDB, SQLite, spaCy, Selenium, RAGAS.
*   **LLM & Embeddings:** `FRIDA-q8_0.gguf` (эмбеддинги), `bge-reranker-v2-m3` (реранкер), `t-lite-it-1.0` (локальная генерация) или OpenRouter API (удалённая генерация по умолчанию).
*   **NER:** `ru_core_news_md` для извлечения ключевых терминов и усиления BM25-поиска.
*   **Микросервисная архитектура:** 3 контейнера (`rag-core`, `web-api`, `frontend`) + серверы llama.cpp (embeddings, reranker, generator).
*   **CPU-friendly:** Все тяжёлые модели вынесены в отдельные Docker-контейнеры с квантованием (Q8_0 / Q4_K_M), работают без GPU.
*   **Полная настраиваемость:** Через `config.py` и `.env` можно изменить системный промпт, выбрать стратегию чанкирования (`fixed`/`sliding`), настроить пороги расширения контекста, переключиться на локальную генерацию и т.д.

---

### Быстрый старт (Docker Compose)

Рекомендуемый способ развёртывания — Docker Compose. Требуется установленный Docker и Docker Compose.

#### 1. Клонирование и подготовка
```bash
git clone <repository-url>
cd <project-folder>
# Скопируйте пример переменных окружения
cp .env.example .env
# Отредактируйте .env: укажите OPENAI_API_KEY (для RAGAS и удалённой генерации)
nano .env
```

#### 2. Запуск инфраструктуры
По умолчанию поднимаются `rag-core`, `web-api`, `frontend`, а также серверы эмбеддингов и реранкера (`llama-embeddings`, `llama-reranker`):
```bash
docker compose up -d
```
При первом запуске `rag-core` автоматически выполнит скрейпинг сайта КубГУ, чанкирование и индексацию (займёт 10–20 минут). После инициализации API станет доступен.

#### 3. Доступ к сервисам
*   **Frontend (UI):** http://localhost:80
*   **Web API (документация Swagger):** http://localhost:8000/docs
*   **RAG Core API (внутренний):** http://rag-core:8001 (доступен только внутри сети Docker)

#### 4. Запуск с локальной генерацией (опционально)
Если хотите использовать локальную LLM вместо OpenRouter:
```bash
docker compose --profile local-generator up -d
```

---

### Ручной запуск (без Docker)

Если требуется запуск отдельных компонентов вне контейнеров:

#### 1. Установка зависимостей
```bash
cd rag_core
python -m venv .venv
source .venv/bin/activate  # или .venv\Scripts\activate на Windows
pip install -r requirements.txt
python -m spacy download ru_core_news_md
```

#### 2. Сбор данных (ETL)
Сбор информации с сайта [КубГУ](https://www.kubsu.ru/ru/) через Selenium. Требуется установленный Google Chrome.
```bash
python cli.py parse
```

#### 3. Чанкирование
Создание чанков в SQLite. Аргумент `-s` отвечает за стратегию:
*   `fixed` — фиксированные чанки с перекрытием (по умолчанию)
*   `sliding` — сегментация по предложениям
```bash
python cli.py database -s fixed
# или
python cli.py database -s sliding
```

#### 4. Индексация
Создание векторного индекса в ChromaDB через API эмбеддингов (llama.cpp сервер должен быть запущен):
```bash
python cli.py index
```

#### 5. Запуск API
```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

#### 6. Запуск интерактивного чата (CLI)
```bash
python cli.py chat
```

---

### Оценка качества (RAGAS)

Для запуска автоматизированной оценки через метрики RAGAS (faithfulness, answer relevancy, context precision, context recall):

```bash
# Генерация датасета из 50 вопросов + ground truth
python evaluate_ragas.py --stage 1 --num-questions 50 --dataset-path data/evaluation/raw_questions.json

# Прогон RAG и оценка
python evaluate_ragas.py --stage 2 --dataset-path data/evaluation/raw_questions.json --results-path data/evaluation/ragas_results.json

# Полный цикл
python evaluate_ragas.py --stage both --num-questions 50
```

Результаты сохраняются в `data/evaluation/ragas_results.json` со средними метриками и конфигурацией.

---

### Конфигурация

Основные параметры задаются через переменные окружения (`.env`) и `rag_core/config.py`:

| Переменная / Параметр | Описание |
|---|---|
| `G_SYSTEM_PROMPT` | Системный промпт для генерации (в `config.py`) |
| `SELECTED_C_CONFIG` | Конфигурация retrieval (`C_KERNEL`, `C_OVERLAP`, `C_DYNAMIC`) |
| `G_USE_REMOTE_MODEL` | `true` — OpenRouter (по умолчанию), `false` — локальная модель |
| `R_EMBEDDING_API_URL` | URL сервера эмбеддингов (по умолчанию `http://llama-embeddings:8083/v1`) |
| `CE_RERANKER_API_URL` | URL сервера реранкера (по умолчанию `http://llama-reranker:8084/v1`) |
| `OPENAI_API_KEY` | Ключ для OpenRouter (генерация + RAGAS) |

Для смены системного промпта достаточно отредактировать `G_SYSTEM_PROMPT` в `config.py` и перезапустить `rag-core`.
