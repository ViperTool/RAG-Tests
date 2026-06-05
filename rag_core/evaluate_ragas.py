#!/usr/bin/env python3
"""
Двухэтапная оценка RAG-системы с помощью RAGAS

Этап 1: Генерация вопросов и эталонных ответов (ground truth)
Этап 2: Прогон RAG по вопросам и оценка качества
"""

import logging
import argparse
import json
from pathlib import Path
import config
from src.utils.dataset_generator import DatasetGenerator
from src.utils.rag_runner import RAGRunner
from src.utils.ragas_evaluator import RagasEvaluator
from src.services.rag import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def stage1_generate_dataset(
        num_questions: int,
        output_path: str,
        use_remote_model: bool
):
    """
    Этап 1: Генерация вопросов и ground truth
    """
    logger.info("=" * 60)
    logger.info("ЭТАП 1: ГЕНЕРАЦИЯ ДАТАСЕТА")
    logger.info("=" * 60)

    # Инициализация RAG для доступа к корпусу
    rag = RAGEngine(collection_name=config.COLLECTION_NAME)

    # Создание генератора
    generator = DatasetGenerator(
        corpus=rag.corpus,
        metadatas=rag.metadatas
    )

    # Генерация вопросов и ответов
    qa_pairs = generator.generate_questions_and_answers(
        num_questions=num_questions,
        use_remote_model=use_remote_model
    )

    if not qa_pairs:
        logger.error("Не удалось сгенерировать ни одной пары!")
        return None

    # Сохранение
    generator.save_dataset(qa_pairs, output_path)

    logger.info(f"\n✅ Этап 1 завершён. Создано {len(qa_pairs)} пар")
    logger.info(f"📁 Датасет сохранён в: {output_path}")

    return qa_pairs


def stage2_evaluate(
        input_path: str,
        output_path: str,
        collection_name: str
):
    """
    Этап 2: Прогон RAG и оценка
    """
    logger.info("=" * 60)
    logger.info("ЭТАП 2: ПРОГОН RAG И ОЦЕНКА")
    logger.info("=" * 60)

    # Загрузка датасета
    generator = DatasetGenerator([], [])
    qa_pairs = generator.load_dataset(input_path)

    if not qa_pairs:
        logger.error("Датасет пуст!")
        return None

    # Прогон RAG
    logger.info("\n--- Прогон RAG по вопросам ---")
    runner = RAGRunner(collection_name=collection_name)
    results = runner.run_on_dataset(
        qa_pairs=qa_pairs,
        chunking_config=config.SELECTED_C_CONFIG
    )

    # Сохранение результатов прогона
    results_path = Path(output_path).parent / "rag_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Результаты прогона сохранены в {results_path}")

    # Оценка через RAGAS
    logger.info("\n--- Оценка через RAGAS ---")
    evaluator = RagasEvaluator()

    dataset = evaluator.create_dataset_from_results(results)
    scores = evaluator.evaluate(dataset)

    # Вывод результатов
    logger.info("\n" + "=" * 60)
    logger.info("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    logger.info("=" * 60)

    print("\n📊 МЕТРИКИ КАЧЕСТВА:")
    print(f"  Faithfulness (Верность):       {scores['faithfulness']:.4f}")
    print(f"  Answer Relevancy (Ответ):      {scores['answer_relevancy']:.4f}")
    print(f"  Context Precision (Точность):  {scores['context_precision']:.4f}")
    print(f"  Context Recall (Полнота):      {scores['context_recall']:.4f}")

    avg_score = sum(scores.values()) / len(scores)
    print(f"\n  СРЕДНЯЯ ОЦЕНКА:                {avg_score:.4f}")

    # Сохранение итоговых результатов
    final_results = {
        'metrics': scores,
        'average_score': avg_score,
        'num_questions': len(results),
        'config': {
            'chunking_config': config.SELECTED_C_CONFIG,
            'generator_model': config.G_LOCAL_MODEL_NAME if not config.G_USE_REMOTE_MODEL else config.G_REMOTE_MODEL_NAME,
            'embedding_model': config.R_EMBEDDING_MODEL_NAME
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    logger.info(f"\nИтоговые результаты сохранены в {output_path}")

    return scores


def main():
    parser = argparse.ArgumentParser(description="Двухэтапная оценка RAG с помощью RAGAS")

    parser.add_argument(
        "--stage",
        type=str,
        choices=['1', '2', 'both'],
        default='both',
        help="Какой этап выполнять: 1 (генерация), 2 (оценка), both (оба)"
    )

    parser.add_argument(
        "--num-questions",
        type=int,
        default=20,
        help="Количество вопросов для генерации (только для этапа 1)"
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/evaluation/raw_questions.json",
        help="Путь к файлу датасета"
    )

    parser.add_argument(
        "--results-path",
        type=str,
        default="data/evaluation/ragas_results.json",
        help="Путь к файлу результатов"
    )

    parser.add_argument(
        "--collection",
        type=str,
        default=config.COLLECTION_NAME,
        help="Имя коллекции ChromaDB"
    )

    parser.add_argument(
        "--use-remote",
        action='store_true',
        default=True,
        help="Использовать remote модель для генерации ground truth"
    )

    args = parser.parse_args()

    # Создаём папку для результатов
    Path(args.dataset_path).parent.mkdir(parents=True, exist_ok=True)

    if args.stage in ['1', 'both']:
        stage1_generate_dataset(
            num_questions=args.num_questions,
            output_path=args.dataset_path,
            use_remote_model=args.use_remote
        )

    if args.stage in ['2', 'both']:
        stage2_evaluate(
            input_path=args.dataset_path,
            output_path=args.results_path,
            collection_name=args.collection
        )


if __name__ == "__main__":
    main()