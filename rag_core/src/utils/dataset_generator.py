import logging
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import config

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """
    Генерация датасета вопросов и эталонных ответов (ground truth)
    """

    def __init__(self, corpus: List[str], metadatas: List[Dict]):
        self.corpus = corpus
        self.metadatas = metadatas

    def sample_contexts(self, num_samples: int = 30) -> List[Dict[str, Any]]:
        """
        Случайная выборка контекстов из базы для генерации вопросов
        """
        logger.info(f"Выборка {num_samples} контекстов из {len(self.corpus)}")

        if num_samples > len(self.corpus):
            num_samples = len(self.corpus)
            logger.warning(f"Запрошено больше, чем есть. Используем {num_samples}")

        sampled_indices = random.sample(range(len(self.corpus)), num_samples)

        samples = []
        for idx in sampled_indices:
            samples.append({
                'index': idx,
                'context': self.corpus[idx],
                'metadata': self.metadatas[idx]
            })

        return samples

    def generate_questions_and_answers(
            self,
            num_questions: int = 20,
            use_remote_model: bool = True
    ) -> List[Dict[str, str]]:
        """
        Генерация вопросов и эталонных ответов

        Args:
            num_questions: Количество вопросов
            use_remote_model: Использовать remote модель для генерации (рекомендуется)

        Returns:
            Список словарей с ключами: question, ground_truth, context
        """
        logger.info(f"Генерация {num_questions} вопросов и эталонных ответов")

        # Выбираем контексты
        samples = self.sample_contexts(num_questions)

        qa_pairs = []

        for i, sample in enumerate(samples):
            try:
                context = sample['context']

                # Промпт для генерации вопроса и ответа
                generation_prompt = f"""
На основе следующего контекста создай:
1. ОДИН конкретный вопрос, на который можно ответить, используя эту информацию. По возможности вопрос должен относиться к категории типичных вопросов студентов, например информация о преподавателсях, о мероприятиях, приемной комиссии, о внутренних заведениях, о преподаваемых дисциплинах
2. Подробный ответ на этот вопрос, основанный ТОЛЬКО на контексте

Контекст:
{context}

Формат ответа (строго):
ВОПРОС: [твой вопрос]
ОТВЕТ: [твой подробный ответ]
"""

                # Используем модель для генерации
                if use_remote_model and config.G_USE_REMOTE_MODEL:
                    # Remote модель (более качественная)
                    from src.core.generators import GeneratorService
                    generator = GeneratorService()

                    response, _, _ = generator.ask_api(
                        query="Создай вопрос и ответ по контексту",
                        context=context
                    )
                else:
                    # Локальная модель
                    from src.core.generators import GeneratorService
                    generator = GeneratorService()

                    response, _, _ = generator.generate_response(
                        query="Создай вопрос и ответ по контексту",
                        context=context,
                        max_new_tokens=512
                    )

                # Парсим ответ
                question = self._extract_question(response)
                answer = self._extract_answer(response)

                if question and answer:
                    qa_pairs.append({
                        'question': question,
                        'ground_truth': answer,
                        'context': context,
                        'source_index': sample['index']
                    })

                    logger.info(f"Сгенерирована пара {i + 1}/{num_questions}")
                    logger.debug(f"Вопрос: {question[:50]}...")
                else:
                    logger.warning(f"Не удалось распарсить ответ для образца {i + 1}")

            except Exception as e:
                logger.error(f"Ошибка при генерации пары {i + 1}: {e}")
                continue

        logger.info(f"Успешно сгенерировано {len(qa_pairs)} пар вопрос-ответ")
        return qa_pairs

    def _extract_question(self, response: str) -> str:
        """Извлекает вопрос из ответа модели"""
        lines = response.strip().split('\n')

        for line in lines:
            if line.startswith('ВОПРОС:'):
                return line.replace('ВОПРОС:', '').strip()
            elif line.startswith('Вопрос:'):
                return line.replace('Вопрос:', '').strip()

        # Если не нашли маркер, берём первую строку
        return lines[0].strip() if lines else ""

    def _extract_answer(self, response: str) -> str:
        """Извлекает ответ из ответа модели"""
        lines = response.strip().split('\n')

        answer_lines = []
        in_answer = False

        for line in lines:
            if line.startswith('ОТВЕТ:') or line.startswith('Ответ:'):
                in_answer = True
                answer_text = line.replace('ОТВЕТ:', '').replace('Ответ:', '').strip()
                if answer_text:
                    answer_lines.append(answer_text)
            elif in_answer:
                answer_lines.append(line)

        return '\n'.join(answer_lines).strip() if answer_lines else ""

    def save_dataset(self, qa_pairs: List[Dict], output_path: str):
        """Сохраняет датасет в JSON"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

        logger.info(f"Датасет сохранён в {output_file}")

    def load_dataset(self, input_path: str) -> List[Dict]:
        """Загружает датасет из JSON"""
        input_file = Path(input_path)

        if not input_file.exists():
            raise FileNotFoundError(f"Датасет не найден: {input_path}")

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Загружено {len(data)} пар из {input_file}")
        return data