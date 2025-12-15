import torch
import requests
import gc
import os
import logging.config
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

from src.utils.wrappers import log_execution
from src.utils import exceptions
import config

logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def form_prompt(query: str, context: str, system_prompt: str = str(config.G_SYSTEM_PROMPT)) -> str:
    return f"Роль: {system_prompt}\n\nКонтекст:\n\n{context}\n\nВопрос:\n\n{query}\n\nОтвет:"


class GeneratorService:
    def __init__(self):
        self.device = config.G_DEVICE
        self.tokenizer = None
        self.model = None
        load_dotenv()

    def load(self, model_name: str = config.G_MODEL_NAME):
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
            )
            self.model.eval()
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

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Просто декодируем ответ без хардкода токенов мыслей
            input_len = model_inputs.input_ids.shape[1]
            output_ids = generated_ids[0][input_len:]

            response_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            return response_text

        except torch.cuda.OutOfMemoryError as e:
            logger.error("OOM во время генерации. Пробуем очистить кэш.")
            torch.cuda.empty_cache()
            raise exceptions.GenerationError("Не хватило памяти для генерации ответа.") from e
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            raise exceptions.GenerationError(f"Сбой inference: {e}") from e

    @staticmethod
    def ask_api(query: str, context: str) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('G_REMOTE_MODEL_API')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": str(config.G_REMOTE_MODEL_NAME),
            "messages": [
                {"role": "system", "content": str(config.G_SYSTEM_PROMPT)},
                {"role": "user", "content": form_prompt(query, context, '')},
            ],
            "temperature": 0.0,
        }
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content')
                if content:
                    logger.info(f"Ответ модели: {content}")
                    return content
                else:
                    logger.error("API вернул пустой ответ")
                    raise exceptions.APIError("API вернул пустой ответ.")
            elif response.status_code == 401:
                logger.error("Ошибка авторизации API. (скорее всего, проблема в ключе)")
                raise exceptions.APIError("Ошибка авторизации API. (скорее всего, проблема в ключе)")
            elif response.status_code == 429:
                logger.error("Превышен лимит запросов API. (повторите попытку через некоторое время)")
                raise exceptions.APIError("Превышен лимит запросов API. (повторите попытку через некоторое время)")
            else:
                logger.error(f"Ошибка API: {response.status_code}")
                raise exceptions.APIError(f"Ошибка API: {response.status_code}")
        except requests.Timeout:
            logger.error("Таймаут соединения с OpenRouter")
            raise exceptions.APIError("Таймаут соединения с OpenRouter")
        except requests.RequestException as e:
            logger.error(f"Сетевая ошибка API: {e}")
            raise exceptions.APIError(f"Сетевая ошибка API: {e}")