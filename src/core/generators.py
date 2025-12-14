from src.utils import config

from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import torch
import requests
import gc
import os

from src.utils.wrappers import log_execution

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
        print(f"Загрузка модели {model_name} на {self.device.upper()}...")

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

    def unload(self):
        """
        Принудительная выгрузка модели из VRAM
        """
        print("Выгрузка модели генератора из VRAM...")

        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

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

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Ошибка OpenRouter: {response.status_code} — {response.text}"
