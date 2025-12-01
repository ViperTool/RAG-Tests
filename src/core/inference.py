import config

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from src.utils.wrappers import log_execution


class GeneratorInferencer:
    def __init__(
        self,
        # base_model_name: str = "t-tech/T-lite-it-1.0",
        base_model_name: str = str(config.G_MODEL_NAME),
        device: str = "cuda",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=(
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            ),
            device_map=device,
            trust_remote_code=True
        )

        self.model.eval()

    @log_execution
    def generate_response(
            self,
            query: str,
            context: str,
            system_prompt: str = str(config.G_SYSTEM_PROMPT),
            max_new_tokens: int = 1024,
    ):

        full_prompt = f"{config.G_SYSTEM_PROMPT}\n\nКонтекст:\n\n{context}\n\nВопрос:\n\n{query}\n\nОтвет:"
        messages = [
            {"role": "user", "content": full_prompt},
        ]

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

        return {
            "question": system_prompt,
            "thinking": "",  # Если мысли нужны, их надо парсить иначе, но пока лучше стабильность
            "response": response_text
        }
