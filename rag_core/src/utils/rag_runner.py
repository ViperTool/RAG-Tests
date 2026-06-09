import logging
from typing import List, Dict, Any
from src.services.rag import RAGEngine
import config

logger = logging.getLogger(__name__)


class RAGRunner:
    """
    Прогон RAG-системы по готовому датасету вопросов
    """

    def __init__(self, collection_name: str):
        logger.info("Инициализация RAG для прогона по датасету")
        self.rag = RAGEngine(collection_name=collection_name)

    def run_on_dataset(
            self,
            qa_pairs: List[Dict[str, str]],
            chunking_config: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Прогон RAG по каждому вопросу из датасета

        Args:
            qa_pairs: Список вопросов и ground truth
            chunking_config: Конфигурация чанкинга

        Returns:
            Список с добавленными полями: answer, contexts
        """
        if chunking_config is None:
            chunking_config = config.SELECTED_C_CONFIG

        logger.info(f"Прогон RAG по {len(qa_pairs)} вопросам")

        results = []

        for i, qa_pair in enumerate(qa_pairs):
            try:
                question = qa_pair['question']

                logger.info(f"Обработка вопроса {i + 1}/{len(qa_pairs)}: {question[:50]}...")

                # 1. Получаем контекст через RAG
                context_text = self.rag.search(
                    query=question,
                    chunkung_config=chunking_config
                )

                # Разбиваем контекст на чанки для RAGAS
                context_chunks = [
                    chunk.strip()
                    for chunk in context_text.split('\n')
                    if chunk.strip()
                ]

                # 2. Генерируем ответ через RAG
                if config.G_USE_REMOTE_MODEL:
                    answer, prompt_tokens, completion_tokens = self.rag.generator_service.ask_api(
                        query=question,
                        context=context_text
                    )
                else:
                    answer, prompt_tokens, completion_tokens = self.rag.generator_service.generate_response(
                        query=question,
                        context=context_text,
                        max_new_tokens=1024
                    )

                # 3. Формируем результат
                result = {
                    **qa_pair,  # Сохраняем question и ground_truth
                    'answer': answer,
                    'contexts': context_chunks,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens
                }

                results.append(result)

                logger.info(f"Вопрос {i + 1} обработан успешно")

            except Exception as e:
                logger.error(f"Ошибка при обработке вопроса {i + 1}: {e}")

                # Добавляем пустой результат, чтобы не ломать порядок
                results.append({
                    **qa_pair,
                    'answer': f"Ошибка: {str(e)}",
                    'contexts': [],
                    'prompt_tokens': 0,
                    'completion_tokens': 0
                })

        logger.info(f"Обработано {len(results)} вопросов")
        return results