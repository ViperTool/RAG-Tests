import argparse
import sys

def cmd_parse(args):
    """Команда: Спарсить данные с вики"""
    from src.utils.parser import run_parsing_pipeline

    run_parsing_pipeline()


def cmd_index(args):
    """Команда: Создать векторный индекс"""
    from src.chroma.chroma_handler import run_chroma_pipeline, delete_collection

    if args.clean:
        delete_collection(args.collection)

    run_chroma_pipeline(collection_name=args.collection)


def cmd_chat(args):
    """Команда: Запустить чат в консоли"""
    from main import main as run_chat
    run_chat()


def main():
    parser = argparse.ArgumentParser(description="Outer Wilds RAG Management Tool")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Доступные команды")

    # Команда: parse
    p_parse = subparsers.add_parser("parse", help="Спарсить данные с Outer Wilds Wiki в SQLite")

    # Команда: index
    p_index = subparsers.add_parser("index", help="Векторизовать данные из SQLite в ChromaDB")
    p_index.add_argument("--clean", action="store_true", help="Удалить старую коллекцию перед созданием")
    p_index.add_argument("--collection", default="wiki_chunks", help="Имя коллекции Chroma")

    # Команда: chat
    p_chat = subparsers.add_parser("chat", help="Запустить интерактивный чат")

    # Парсинг аргументов
    args = parser.parse_args()

    # Запуск соответствующей функции
    if args.command == "parse":
        cmd_parse(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "chat":
        cmd_chat(args)


if __name__ == "__main__":
    main()
