#!/bin/bash
set -e

# Проверяем, существует ли файл базы SQLite или папка ChromaDB
if [ ! -f "/app/data/chroma.sqlite3" ] || [ ! -f "/app/data/wiki_content.db" ]; then
    echo "База данных не найдена. Начинаем сбор и индексацию данных..."
    # Здесь вызываем ваш CLI или скрипт парсинга
    python cli.py parse
    echo "DEBUG: Модель эмбеддингов = $R_EMBEDDING_MODEL_NAME"
    python cli.py database -s fixed
    python cli.py index
    echo "База данных успешно создана."
else
    echo "База данных найдена. Пропуск инициализации."
fi

# Передаем управление команде из Dockerfile (uvicorn)
exec "$@"
