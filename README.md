# Outer Wilds RAG Assistant

Интеллектуальный ассистент по базе знаний игры *Outer Wilds*, построенный на архитектуре RAG (Retrieval Augmented Generation).

Проект реализует **гибридный поиск** (BM25 + Embeddings) и кастомный алгоритм **Context Reranking** на основе Sentence Window Retrieval с применением гауссовского сглаживания (KDE) для восстановления контекста.

### Особенности
*   **Стек:** Python 3.10+, PyTorch, Transformers, ChromaDB, SQLite, BeautifulSoup4.
*   **LLM & Embeddings:** Использует `Qwen3-1.7B` (генерация), `ai-forever/FRIDA` (эмбеддинги) и `ru_core_news_md` (Задачи NER).
*   **Оптимизация памяти:** Раздельная загрузка моделей (CPU/GPU) для запуска на картах с небольшим количеством памяти.
*   **Умный поиск:** Собственная реализация ранжирования чанков, объединяющая лексический поиск с NER и семантический поиск.
*   **Полная настраиваемость**: При желании можно выбрать другую Вики, другую модель для генерации и эмбеддингов, настроить названия коллекций, изменить системный промпт и так далее с помощью понятного config.py.

---

### Установка и запуск

#### 1. Клонирование и подготовка
git clone https://github.com/thererealareyou/RAG-Tests.git

cd RAG-tests

Создание и активация виртуального окружения (Windows)

python -m venv .venv
.\\.venv\Scripts\activate

Установка зависимостей

pip install -r requirements.txt

#### 2. Сбор данных (ETL)
Сбор информации с [Outer Wilds Wiki](https://outer-wilds.fandom.com/ru/wiki/) и сохранение в SQLite:

python cli.py parse

#### 3. Индексация
Создание векторного индекса в ChromaDB. Используйте флаг `--clean` для полной пересборки базы.

python cli.py index --clean

#### 4. Запуск чата
python cli.py chat

---

### Конфигурация
Основные настройки (пути, модели, параметры поиска) находятся в файле `config.py`.