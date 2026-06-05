import requests
import logging
import os
from typing import Tuple, Any
from src.utils.wrappers import log_execution
from src.utils import exceptions
import config

logger = logging.getLogger(__name__)


def form_prompt(query: str, context: str, system_prompt: str = str(config.G_SYSTEM_PROMPT)) -> str:
    prompt = f"Роль: {system_prompt}\n\n" if system_prompt else ""
    return prompt + f"Контекст:\n\n{context}\n\nВопрос:\n\n{query}\n\nОтвет:"


class GeneratorService:
    def __init__(self):
        self.base_url = config.G_LOCAL_MODEL_URL

    @log_execution
    def generate_response(
            self,
            query: str,
            context: str,
            max_new_tokens: int = 1024,
    ) -> Tuple[str, int, int]:
        """
        Теперь это запрос к локальному llama server (OpenAI-compatible API)
        """
        url = f"{self.base_url}/chat/completions"

        user_content = form_prompt(query, context)

        payload = {
            "model": config.G_LOCAL_MODEL_NAME,
            "messages": [
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens": max_new_tokens
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()

            data = response.json()
            content = data['choices'][0]['message']['content'].strip()

            usage = data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)

            logger.info(f"Локальный сервер ответил успешно. Токены: {prompt_tokens} in / {completion_tokens} out")
            return content, prompt_tokens, completion_tokens

        except Exception as e:
            logger.error(f"Ошибка при обращении к локальному Llama серверу: {e}")
            raise exceptions.GenerationError(f"Llama server error: {e}")

    @staticmethod
    def ask_api(query: str, context: str) -> Tuple[Any, Any, Any]:
        url = "https://openrouter.ai/api/v1/chat/completions"

        api_key = os.getenv('G_REMOTE_MODEL_API')
        if not api_key:
            logger.critical("Не найден API ключ в переменных окружения (G_REMOTE_MODEL_API)")
            raise exceptions.APIError("API Key is missing")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            user_content = form_prompt(query, context, '')
        except NameError:
            logger.error("Функция form_prompt не найдена!")
            raise
        except Exception as e:
            logger.error(f"Ошибка при формировании промпта: {e}")
            raise

        payload = {
            "model": str(config.G_REMOTE_MODEL_NAME),
            "messages": [
                {"role": "system", "content": str(config.G_SYSTEM_PROMPT)},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                choice = data.get('choices', [{}])[0]
                content = choice.get('message', {}).get('content')

                usage = data.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)

                if content:
                    logger.info(f"Успех. Ответ: {content}\n\nТокены: {prompt_tokens} (in) / {completion_tokens} (out)")
                    return content, prompt_tokens, completion_tokens
                else:
                    logger.error(f"Пустой ответ API. Raw response: {data}")
                    raise exceptions.APIError("API вернул корректный JSON, но без контента.")

            else:
                error_msg = response.text
                logger.error(f"API Error {response.status_code}: {error_msg}")

                if response.status_code == 401:
                    raise exceptions.APIError("Ошибка 401: Неверный API ключ.")
                elif response.status_code == 429:
                    raise exceptions.APIError("Ошибка 429: Лимит запросов или баланс исчерпан.")
                else:
                    raise exceptions.APIError(f"Ошибка API {response.status_code}: {error_msg}")

        except requests.Timeout:
            logger.error("Таймаут соединения с OpenRouter (более 30 сек)")
            raise exceptions.APIError("Таймаут соединения")
        except requests.RequestException as e:
            logger.error(f"Сетевая ошибка: {e}")
            raise exceptions.APIError(f"Сетевая ошибка: {e}")