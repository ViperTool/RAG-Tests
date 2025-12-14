import time
import re
import urllib.parse
import requests
import logging, logging.config
from bs4 import BeautifulSoup
from typing import List, Optional
from tqdm import tqdm

from src.sqlite.sqlite_handler import SQLiteManager
from src.utils import config, exceptions

logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class WikiScraper:
    def __init__(self, base_url: str = str(config.WIKI_URL)):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        })

    def get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """
        Скачивает страницу и возвращает объект BeautifulSoup с повторными попытками.
        """
        retries = 3
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 429:
                    logger.error(f"Ошибка 429 {url}. Ожидаем.")
                    time.sleep(5)
                    continue
                response.raise_for_status()
                return BeautifulSoup(response.content, "html.parser")
            except requests.RequestException as e:
                logger.error(f"Ошибка при загрузке {url} (попытка {attempt + 1}): {e}")
                time.sleep(2)
        logger.critical(f"Сбой загрузки {url} после {retries} попыток.")
        raise exceptions.NetworkError(f"Сбой загрузки: {url}")

    def get_all_page_links(self) -> List[str]:
        """
        Собирает ссылки на ВСЕ страницы вики через служебную страницу.
        """
        start_url = urllib.parse.urljoin(self.base_url, "Служебная:Все_страницы")
        urls = []
        current_url = start_url

        logger.info(f"Начинаем сбор ссылок на страницы")
        while current_url:
            try:
                soup = self.get_soup(current_url)
                if not soup:
                    break

                body = soup.find('div', class_='mw-allpages-body')
                if body:
                    links = body.find_all('a')
                    for link in links:
                        href = link.get('href')
                        if href:
                            full_url = urllib.parse.urljoin(self.base_url, href)
                            urls.append(full_url)

                nav = soup.find('div', class_='mw-allpages-nav')
                next_link = None
                if nav:
                    for a in nav.find_all('a'):
                        if "Следующая страница" in a.text or "Next page" in a.text or ">" in a.text:
                            next_link = a
                            break

                if next_link:
                    current_url = urllib.parse.urljoin(self.base_url, next_link.get('href'))
                    logger.info(f"Переход к следующей странице списка: {current_url}")
                else:
                    current_url = None
            except exceptions.NetworkError as e:
                logger.error(f"Прерываю парсинг страниц из-за ошибки сети. {e}")
                break

        unique_urls = list(set(urls))
        logger.info(f"Всего найдено ссылок: {len(unique_urls)}")
        return unique_urls

    def parse_page(self, url: str) -> Optional[str]:
        """
        Парсинг страницы вики, возвращает контент страницы.

        Args:
             url (str): URL страницы.
        """
        try:
            soup = self.get_soup(url)
            if not soup:
                return None

            title_elem = soup.find('span', class_='mw-page-title-main')
            title = title_elem.text.strip() if title_elem else "No Title"

            content_div = soup.find('div', class_='mw-content-ltr')
            if not content_div:
                return None

            raw_text = content_div.get_text(separator=' ', strip=True)

            cleaned_text = self.clean_text(raw_text)

            return f"{title}\n\n{cleaned_text}"
        except Exception as e:
            logger.critical(f"Ошибка при парсинге страницы {url}: {e}")
            raise exceptions.ParsingError(f"Ошибка при парсинге страницы {url}: {e}")

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Специфичная для Outer Wilds очистка текста.
        """
        trash_phrases = [
            '(Издания для ПК, консолей, консолей старого поколения и мобильных устройств)',
            'ВНИМАНИЕ, СПОЙЛЕРЫ : Статья содержит детали сюжета игры Outer Wilds',
            'View or edit this template'
        ]

        for phrase in trash_phrases:
            text = text.replace(phrase, '')

        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()


    def run_scraping_pipeline(self):
        """
        Парсит выбранную вики и сохраняет её в базу данных SQLite
        """
        logger.info("ЗАПУСК: Пайплайн скрапера вики.")

        db_manager = SQLiteManager()
        db_manager.create_schema()
        sleep_time = 5

        try:
            links = self.get_all_page_links()
        except exceptions.NetworkError as e:
            logger.critical(f"Не удалось собрать ссылки: {e}")
            return

        logger.info(f"Начинаем загрузку {len(links)} страниц...")

        for url in tqdm(links, desc="Downloading pages"):
            try:
                content = self.parse_page(url)

                if content:
                    db_manager.save_or_update_page_by_url(url, content)

                time.sleep(sleep_time)

            except exceptions.NetworkError:
                logger.error(f"Пропуск {url} из-за ошибки сети")
                continue
            except exceptions.ParsingError:
                logger.error(f"Пропуск {url} из-за ошибки парсинга")
                continue
            except Exception as e:
                logger.critical(f"Непредвиденная ошибка на {url}: {e}")

        logger.info("КОНЕЦ: Пайплайн скрапера вики.")

if __name__ == "__main__":
    wiki_scraper = WikiScraper()
    wiki_scraper.run_scraping_pipeline()