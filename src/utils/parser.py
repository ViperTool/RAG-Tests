import time
import re
import urllib.parse
import requests
from bs4 import BeautifulSoup
from typing import List, Optional
from tqdm import tqdm


import config
import src.sqlite.sqlite_handler as db_handler


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
                    print(f"Ошибка 429 {url}. Ожидаем.")
                    time.sleep(5)
                    continue
                response.raise_for_status()
                return BeautifulSoup(response.content, "html.parser")
            except requests.RequestException as e:
                print(f"Ошибка при загрузке {url} (попытка {attempt + 1}): {e}")
                time.sleep(2)
        return None

    def get_all_page_links(self) -> List[str]:
        """
        Собирает ссылки на ВСЕ страницы вики через служебную страницу.
        """
        start_url = urllib.parse.urljoin(self.base_url, "Служебная:Все_страницы")
        urls = []
        current_url = start_url

        print("Начинаем сбор ссылок на страницы...")
        while current_url:
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
                print(f"Переход к следующей странице списка: {current_url}")
            else:
                current_url = None

        print(f"Всего найдено ссылок: {len(urls)}")
        return list(set(urls))

    def parse_page(self, url: str) -> Optional[str]:
        """
        Парсинг страницы вики, возвращает контент страницы.

        Args:
             url (str): URL страницы.
        """
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


def run_parsing_pipeline():
    """
    Основная функция, которая запускает парсинг и сохранение в БД.
    """
    scraper = WikiScraper()

    db_handler.create_content()

    links = scraper.get_all_page_links()

    print(f"Начинаем парсинг {len(links)} страниц...")

    for url in tqdm(links):
        existing = db_handler.get_page_by_url(url)
        if existing:
            continue

        content = scraper.parse_page(url)
        if content:
            page_id = db_handler.save_or_update_page_by_url(url, content)

            title = content.split('\n')[0]
            chunks = db_handler.form_chunks_swr(title, content)

            if page_id:
                db_handler.save_chunks(page_id, chunks)

        time.sleep(5)
