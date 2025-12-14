import argparse
import sys

from src.utils.parser import WikiScraper
from src.chroma.chroma_handler import ChromaManager
from main import main as run_chat


def cmd_parse(args):
    """Команда: Спарсить данные с вики"""
    wiki_parser = WikiScraper()
    wiki_parser.run_scraping_pipeline()


def cmd_index(args):
    """Команда: Создать векторный индекс"""
    chroma_manager = ChromaManager()
    if args.clean:
        chroma_manager.delete_collection(args.collection)

    chroma_manager.run_chroma_pipeline(collection_name=args.collection)


def cmd_chat(args):
    """Команда: Запустить чат в консоли"""
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
