import time
import re
import urllib.parse
import logging.config
from typing import List, Optional

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from bs4 import BeautifulSoup
from tqdm import tqdm

from src.sqlite.sqlite_handler import SQLiteManager
from src.utils import exceptions
from src.utils.logger import init_logging
import config

logger = logging.getLogger(__name__)


class WikiScraper:
    def __init__(self, base_url: str = str(config.WIKI_URL)):
        logger.info("Инициализация экземпляра класса WikiScraper")
        self.base_url = base_url
        self.driver = self._init_driver()

    @staticmethod
    def _init_driver():
        """Настройка и запуск Chrome."""
        logger.info("Запуск браузера Selenium...")
        chrome_options = Options()

        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

        chrome_options.page_load_strategy = 'eager'

        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)

            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            return driver
        except Exception as e:
            logger.critical(f"Не удалось запустить Selenium: {e}")
            raise exceptions.NetworkError("Ошибка инициализации драйвера") from e

    def _close(self):
        """Закрывает браузер и освобождает ресурсы."""
        if self.driver:
            logger.info("Закрытие браузера...")
            self.driver.quit()

    def _get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """
        Загружает страницу через Selenium и возвращает BeautifulSoup.
        """
        retries = 3
        for attempt in range(retries):
            try:
                logger.debug(f"Переход на {url}...")
                self.driver.get(url)

                time.sleep(3)

                html = self.driver.page_source
                return BeautifulSoup(html, "html.parser")

            except Exception as e:
                logger.warning(f"Ошибка Selenium при загрузке {url} (попытка {attempt + 1}): {e}")
                time.sleep(5)

        logger.error(f"Сбой загрузки {url} после {retries} попыток.")
        raise exceptions.NetworkError(f"Сбой загрузки: {url}")

    def _get_all_page_links(self) -> List[str]:
        """
        Собирает ссылки на ВСЕ страницы вики через служебную страницу.
        """
        start_url = urllib.parse.urljoin(self.base_url, "Служебная:Все_страницы")
        urls = []
        current_url = start_url

        logger.info(f"Начинаем сбор ссылок (стартовая: {start_url})")

        while current_url:
            try:
                soup = self._get_soup(current_url)
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
                else:
                    logger.warning(f"Не найден контейнер mw-allpages-body на {current_url}")

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
                logger.error(f"Прерываю сбор ссылок: {e}")
                break

        unique_urls = list(set(urls))
        logger.info(f"Всего найдено ссылок: {len(unique_urls)}")
        return unique_urls

    def _parse_page(self, url: str) -> Optional[str]:
        try:
            soup = self._get_soup(url)
            if not soup:
                return None

            title_elem = soup.find('span', class_='mw-page-title-main')
            title = title_elem.text.strip() if title_elem else "No Title"

            content_div = soup.find('div', class_='mw-content-ltr')
            if not content_div:
                logger.warning(f"Контент не найден: {url}")
                return None

            raw_text = content_div.get_text(separator=' ', strip=True)
            cleaned_text = self._clean_text(raw_text)

            return f"{title}\n\n{cleaned_text}"
        except Exception as e:
            logger.error(f"Ошибка при парсинге {url}: {e}")
            raise exceptions.ParsingError(f"Ошибка парсинга: {url}") from e

    @staticmethod
    def _clean_text(text: str) -> str:
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
        logger.info("ЗАПУСК: Пайплайн скрапера вики.")
        db_manager = SQLiteManager()

        try:
            db_manager.create_schema()

            links = self._get_all_page_links()

            if not links:
                logger.warning("Ссылок не найдено. Проверьте адрес вики.")
                return

            logger.info(f"Начинаем загрузку {len(links)} страниц...")

            for url in tqdm(links, desc="Скачивание страниц: "):
                try:
                    print(url)
                    content = self._parse_page(url)
                    if content:
                        db_manager.save_or_update_page_by_url(url, content)

                    time.sleep(2)

                except exceptions.NetworkError:
                    continue
                except exceptions.ParsingError:
                    continue
                except Exception as e:
                    logger.error(f"Непредвиденная ошибка на {url}: {e}")

        except Exception as e:
            logger.critical(f"Фатальная ошибка пайплайна: {e}")
        finally:
            self._close()
            logger.info("КОНЕЦ: Пайплайн завершен.")


if __name__ == "__main__":
    init_logging()
    wiki_scraper = WikiScraper()
    wiki_scraper.run_scraping_pipeline()
