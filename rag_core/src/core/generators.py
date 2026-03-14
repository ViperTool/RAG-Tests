from typing import Any, Tuple

import torch
import requests
import gc
import os
import logging.config
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

from rag_core.src.utils.wrappers import log_execution
from rag_core.src.utils import exceptions
import config

logger = logging.getLogger(__name__)

def form_prompt(query: str, context: str, system_prompt: str = str(config.G_SYSTEM_PROMPT)) -> str:
    prompt = f"Роль: {system_prompt}\n\n" if system_prompt else ""
    return prompt + f"Контекст:\n\n{context}\n\nВопрос:\n\n{query}\n\nОтвет:"


class GeneratorService:
    def __init__(self):
        self.device = config.G_DEVICE
        self.tokenizer = None
        self.model = None
        load_dotenv()

    def load(self, model_name: str = config.G_LOCAL_MODEL_NAME):
        """
        Загрузка модели
        """
        logger.info(f"Загрузка генеративной модели {model_name} на {self.device.upper()}...")

        if self.model is not None:
            logger.info("Генеративная модель уже загружена.")
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=(
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                ),
                device_map=self.device,
                trust_remote_code=True
            ).eval()
            logger.info("Генеративная модель успешно загружена.")

        except OSError as e:
            logger.critical(f"Не удалось найти или скачать генеративную модель {model_name}: {e}")
            raise exceptions.ModelLoadingError(f"Ошибка файлов генеративной модели: {e}") from e
        except torch.cuda.OutOfMemoryError as e:
            logger.critical(f"Недостаточно VRAM для загрузки генеративной модели {model_name}")
            self.unload()
            raise exceptions.ModelLoadingError(f"Недостаточно VRAM для загрузки генеративной модели {model_name}") from e
        except Exception as e:
            logger.critical(f"Ошибка инициализации GeneratorService: {e}")
            raise exceptions.ModelLoadingError(f"Неизвестная ошибка инициализации GeneratorService: {e}") from e

    def unload(self):
        """
        Принудительная выгрузка модели из VRAM
        """
        logger.info("Выгрузка генеративной модели из VRAM...")
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
        except Exception as e:
            logger.warning(f"Ошибка при удалении объектов генеративной модели: {e}")
        finally:
            self.model = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Память очищена")

    @log_execution
    def generate_response(
            self,
            query: str,
            context: str,
            max_new_tokens: int = 1024,
    ):

        full_prompt = form_prompt(query, context)
        messages = [
            {"role": "user", "content": full_prompt},
        ]

        try:
            if not self.model:
                self.load()

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            prompt_tokens = model_inputs.input_ids.shape[1]

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            total_ids_count = generated_ids.shape[1]
            completion_tokens = total_ids_count - prompt_tokens

            output_ids = generated_ids[0][prompt_tokens:]
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            logger.info(f"Ответ модели: {content}\n"
                        f"Количество токенов промпта: {prompt_tokens}\n"
                        f"Количество использованных токенов: {completion_tokens}")

            return content, prompt_tokens, completion_tokens


        except torch.cuda.OutOfMemoryError as e:
            logger.error("OOM во время генерации. Пробуем очистить кэш.")
            torch.cuda.empty_cache()
            raise exceptions.GenerationError("Не хватило памяти для генерации ответа.") from e
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            raise exceptions.GenerationError(f"Сбой inference: {e}") from e

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