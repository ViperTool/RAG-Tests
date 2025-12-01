import config

from src.services.rag import RAGEngine
from src.core.inference import GeneratorInferencer


def main():
    rag = RAGEngine(collection_name=config.COLLECTION_NAME)
    inferencer = GeneratorInferencer()

    print("\n=== Система запущена. ===")

    while True:
        query = input("\nВопрос: ")
        context = rag.search(query)
        print(context)

        result = inferencer.generate_response(
            query=query, context=context
        )
        print(f"\nОтвет:\n{result['response']}")


if __name__ == "__main__":
    main()
