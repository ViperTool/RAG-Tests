import config

from src.services.rag import RAGEngine

def main():
    rag = RAGEngine(collection_name=config.COLLECTION_NAME)

    print("\n=== Система запущена. ===")

    while True:
        query = input("\nВопрос: ")
        context = rag.search(query)
        print(context)

        result = rag.generator_service.ask_api(query=query,
                                             context=context) \
            if config.G_USE_REMOTE_MODEL \
            else rag.generator_service.generate_response(query=query,
                                              context=context)

        print(f"\nОтвет:\n{result}")

if __name__ == "__main__":
    main()
