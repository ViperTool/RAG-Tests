import argparse
import sys

from src.utils.parser_ow import WikiScraper
from src.chroma.chroma_handler import ChromaManager
from src.sqlite.sqlite_handler import SQLiteManager
from main import main as run_chat


def cmd_parse(args):
    """Команда: Спарсить данные с вики"""
    wiki_parser = WikiScraper()
    wiki_parser.run_scraping_pipeline()

def cmd_database(args):
    """Команда: Сформировать базу данных"""
    sqlite_manager = SQLiteManager()
    sqlite_manager.run_processing_pipeline(strategy=args.strategy)

def cmd_index(args):
    """Команда: Создать векторный индекс"""
    chroma_manager = ChromaManager()
    chroma_manager.run_chroma_pipeline()

def cmd_chat(args):
    """Команда: Запустить чат в консоли"""
    run_chat()

def main():
    parser = argparse.ArgumentParser(description="Outer Wilds RAG Management Tool")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Доступные команды")

    p_parse = subparsers.add_parser("parse", help="Спарсить данные с Outer Wilds Wiki в SQLite")
    p_database = subparsers.add_parser("database", help="Создать чанки и сохранить их в SQLite")
    p_database.add_argument(
        "--strategy", "-s",
        type=str,
        default="fixed",  # Значение по умолчанию
        choices=["fixed", "sliding"],  # Ограничиваем выбор только этими вариантами
        help="Стратегия чанкирования: 'fixed' (фиксированная с перекрытием) или 'sliding' (плавающее окно)"
    )
    p_index = subparsers.add_parser("index", help="Векторизовать данные из SQLite в ChromaDB")
    p_chat = subparsers.add_parser("chat", help="Запустить интерактивный чат")

    args = parser.parse_args()

    if args.command == "parse":
        cmd_parse(args)
    elif args.command == "database":
        cmd_database(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "chat":
        cmd_chat(args)


if __name__ == "__main__":
    main()
