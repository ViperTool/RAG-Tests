FROM python:3.10-slim

WORKDIR /app

# Копирование requirements и установка зависимостей
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Загрузка spaCy модели
RUN python -m spacy download ru_core_news_md

# Копирование кода
COPY . .

# Создание необходимых директорий
RUN mkdir -p /app/data /app/logs

# Запуск приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
