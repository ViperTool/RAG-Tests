import re
import logging
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import config
import os
import numpy as np

logger = logging.getLogger(__name__)


class RagasEvaluator:
    """
    Оценка качества RAG с помощью RAGAS с детальной диагностикой
    """

    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        # Настраиваем LLM для RAGAS
        self.llm = self._setup_llm()

        # КРИТИЧНО: Настраиваем embeddings для русского языка
        self.embeddings = self._setup_embeddings()

        logger.info("RAGAS evaluator инициализирован")

    def _setup_llm(self):
        """Настраивает LLM для оценки метрик RAGAS"""
        api_key = os.getenv("G_REMOTE_MODEL_API_KEY") or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "Не найден API ключ для RAGAS. "
                "Установите переменную окружения G_REMOTE_MODEL_API_KEY или OPENAI_API_KEY"
            )

        base_url = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

        llm = ChatOpenAI(
            model=config.G_REMOTE_MODEL_NAME,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.0,
            max_tokens=1024
        )

        logger.info(f"RAGAS LLM: {config.G_REMOTE_MODEL_NAME} через {base_url}")
        return llm

    def _setup_embeddings(self):
        """
        Настраивает embeddings для RAGAS через существующий EmbeddingService.
        """
        try:
            from src.utils.ragas_embeddings import RagasEmbeddingsAdapter

            # Используем тот же сервис, что и в RAG (или создаём новый)
            embeddings = RagasEmbeddingsAdapter()

            # Тестовая проверка
            test_emb = embeddings.embed_query("тест")
            logger.info(f"Embeddings адаптер создан")
            logger.info(f"Размер вектора: {len(test_emb)}")
            logger.info(f"Первые 5 значений: {test_emb[:5]}")

            return embeddings

        except ImportError as e:
            logger.error(f"Не удалось импортировать RagasEmbeddingsAdapter: {e}")
        except Exception as e:
            logger.error(f"Не удалось создать embeddings адаптер: {e}")
            import traceback
            logger.error(traceback.format_exc())

        logger.warning("RAGAS будет работать без embeddings (answer_relevancy может быть 0)")
        return None

    def _extract_main_answer(self, full_answer: str) -> str:
        """
        Извлекает только основную часть ответа из структурированного текста.
        """
        if not full_answer:
            return ""

        # Попытка 1: Ищем секцию "Ответ:"
        patterns = [
            r'Ответ:\s*(.+?)(?=\n\s*(?:Примечание|###|$))',
            r'ответ:\s*(.+?)(?=\n\s*(?:примечание|###|$))',
        ]

        for pattern in patterns:
            match = re.search(pattern, full_answer, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) > 10:  # Минимальная длина
                    return extracted

        # Попытка 2: Убираем служебные секции
        cleaned = full_answer
        cleaned = re.sub(r'Анализ запроса:.*?\n', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'Примечание:.*$', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = cleaned.strip()

        if len(cleaned) > 10:
            return cleaned

        # Если ничего не помогло, возвращаем как есть
        logger.warning(f"Не удалось очистить ответ, используется оригинал: {full_answer[:100]}...")
        return full_answer

    def _validate_answer(self, answer: str) -> bool:
        """Проверяет, что ответ достаточно хороший для оценки"""
        if not answer or len(answer.strip()) < 10:
            return False

        # Проверяем, что ответ не состоит только из служебных фраз
        error_phrases = [
            "информации для ответа недостаточно",
            "нет информации для ответа",
            "в предоставленных данных нет",
        ]

        answer_lower = answer.lower()
        for phrase in error_phrases:
            if phrase in answer_lower and len(answer) < 100:
                return False

        return True

    def create_dataset_from_results(
            self,
            results: List[Dict[str, Any]]
    ) -> Dataset:
        """
        Создаёт датасет RAGAS с детальной диагностикой
        """
        logger.info("=" * 60)
        logger.info("СОЗДАНИЕ ДАТАСЕТА ДЛЯ RAGAS")
        logger.info("=" * 60)

        questions = []
        answers = []
        contexts = []
        ground_truths = []

        valid_count = 0
        invalid_count = 0

        for i, r in enumerate(results):
            question = r['question']
            raw_answer = r['answer']
            context = r['contexts']
            ground_truth = r['ground_truth']

            # Очистка ответа
            cleaned_answer = self._extract_main_answer(raw_answer)

            # Валидация
            is_valid = self._validate_answer(cleaned_answer)

            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                logger.warning(f"Ответ {i + 1} не прошёл валидацию: {cleaned_answer[:100]}...")

            # Логируем первые 3 примера
            if i < 3:
                logger.info(f"\n--- Пример {i + 1} ---")
                logger.info(f"Вопрос: {question}")
                logger.info(f"Оригинальный ответ: {raw_answer[:150]}...")
                logger.info(f"Очищенный ответ: {cleaned_answer[:150]}...")
                logger.info(f"Валиден: {is_valid}")
                logger.info(f"Контекстов: {len(context)}")

            questions.append(question)
            answers.append(cleaned_answer)
            contexts.append(context)
            ground_truths.append(ground_truth)

        logger.info(f"Валидных ответов: {valid_count}/{len(results)}")
        logger.info(f"Невалидных ответов: {invalid_count}/{len(results)}")

        data = {
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truth': ground_truths
        }

        dataset = Dataset.from_dict(data)
        logger.info(f"Создан датасет RAGAS: {len(questions)} примеров")

        return dataset

    def evaluate(self, dataset: Dataset) -> Dict[str, float]:
        """
        Оценка датасета с детальной диагностикой
        """
        logger.info("\n" + "=" * 60)
        logger.info("НАЧАЛО ОЦЕНКИ RAGAS")
        logger.info("=" * 60)

        logger.info(f"Метрики: {[m.name for m in self.metrics]}")
        logger.info(f"LLM: {self.llm.model_name}")
        logger.info(f"Embeddings: {'Загружены' if self.embeddings else 'Не загружены'}")

        try:
            # Формируем параметры для evaluate
            eval_kwargs = {
                'dataset': dataset,
                'metrics': self.metrics,
                'llm': self.llm,
                'raise_exceptions': False
            }

            # КРИТИЧНО: Передаём embeddings явно
            if self.embeddings:
                eval_kwargs['embeddings'] = self.embeddings
                logger.info("Embeddings переданы в RAGAS")
            else:
                logger.warning("Embeddings не переданы, answer_relevancy может быть 0")

            logger.info("\nЗапуск оценки (это может занять несколько минут)...")
            result = evaluate(**eval_kwargs)

            # Извлекаем результаты
            scores = {}
            for metric_name in ['faithfulness', 'answer_relevancy',
                                'context_precision', 'context_recall']:
                try:
                    value = result[metric_name]

                    if isinstance(value, (int, float)):
                        if np.isnan(value):
                            logger.warning(f"{metric_name}: NaN → 0.0")
                            scores[metric_name] = 0.0
                        else:
                            scores[metric_name] = float(value)
                    else:
                        # Если это массив, берём среднее
                        scores[metric_name] = float(np.mean(value))

                except Exception as e:
                    logger.error(f"Ошибка при извлечении {metric_name}: {e}")
                    scores[metric_name] = 0.0

            logger.info(f"Результаты оценки: {scores}")
            return scores

        except Exception as e:
            logger.error(f"Критическая ошибка при оценке RAGAS: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise