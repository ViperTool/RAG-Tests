from typing import Any, List, Sequence, override
import logging
import torch
from llama_index.core.base.embeddings.base import similarity
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import StoppingCriteriaList, StoppingCriteria

from llama_index.core import Document, VectorStoreIndex, get_response_synthesizer, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole, CompletionResponse
from llama_index.core.llms.callbacks import llm_completion_callback


class SequenceStoppingCriteria(StoppingCriteria):
    def __init__(self, sequences: List[List[int]]):
        super().__init__()
        self.sequence_lists = sequences
        self.seq_tensors = [torch.tensor(seq, dtype=torch.long) for seq in sequences]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        device_of_input = input_ids.device
        current_length = input_ids.shape[-1]
        for i, seq_list in enumerate(self.sequence_lists):
            seq_len = len(seq_list)
            if current_length >= seq_len:
                last_tokens = input_ids[0, -seq_len:]
                seq_tensor_on_device = self.seq_tensors[i].to(device_of_input)
                if torch.equal(last_tokens, seq_tensor_on_device):
                    return True
        return False

class RagSearch:
    def __init__(self):
        self.embeddings = self.FridaEmbeddings()

        Settings.embed_model = self.embeddings
        Settings.node_parser = SentenceSplitter(chunk_size=128, chunk_overlap=20)
        self.target_sequence = [522, 82, 397]

        self.system_prompt = """Ты - интеллектуальный собеседник. Тебе необходимо ответить на вопрос пользователя, основываясь на контекст, предоставленный ниже. В случае, если контекста недостаточно, напиши 'НЕДОСТАТОЧНО ДАННЫХ'"""

        Settings.llm = self.CustomHuggingFaceLLM(
            model_name="Qwen/Qwen3-1.7B",
            tokenizer_name="Qwen/Qwen3-1.7B",
            context_window=1024,
            max_new_tokens=64,
            generate_kwargs={
                "temperature": 0.1,
                "top_p": 0.5,
                "do_sample": True,
                "repetition_penalty": 4.0,
            },
            device_map="cuda",
            system_prompt=self.system_prompt,
            stop_sequences=[self.target_sequence],
        )

        self.client = QdrantClient(path="./local_qdrant")
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name="pages_chunks",
            embed_dimension=768
        )

        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(self.vector_store, embed_model=self.embeddings)
        self.query_engine = self.index.as_query_engine(similarity_top_k=5)

    def ask(self, query: str) -> str:
        response = self.query_engine.query(query)
        print(response.respone,)
        return response.response

    class CustomHuggingFaceLLM(HuggingFaceLLM):
        def __init__(self, stop_sequences=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._stopping_criteria_list = StoppingCriteriaList()
            if stop_sequences:
                self._stopping_criteria_list.append(SequenceStoppingCriteria(stop_sequences))

        @llm_completion_callback()
        def complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> CompletionResponse:
            full_prompt = prompt
            if not formatted:
                if self.query_wrapper_prompt:
                    full_prompt = self.query_wrapper_prompt.format(query_str=prompt)
                if self.completion_to_prompt:
                    full_prompt = self.completion_to_prompt(full_prompt)
                elif self.system_prompt:
                    full_prompt = f"{self.system_prompt} {full_prompt}"

            inputs = self._tokenizer(full_prompt, return_tensors="pt")
            inputs = inputs.to(self._model.device)

            for key in self.tokenizer_outputs_to_remove:
                if key in inputs:
                    inputs.pop(key, None)

            stopping_criteria_to_use = self._stopping_criteria_list
            if "stopping_criteria" in kwargs:
                kwargs_sc = kwargs.pop("stopping_criteria", None)
                if kwargs_sc is not None:
                     stopping_criteria_to_use = stopping_criteria_to_use + kwargs_sc

            tokens = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                stopping_criteria=stopping_criteria_to_use,
                **self.generate_kwargs,
                **kwargs
            )
            completion_tokens = tokens[0][inputs["input_ids"].size(1) :]

            for token_id in completion_tokens.tolist():
                decoded_token = self._tokenizer.decode([token_id], clean_up_tokenization_spaces=True, skip_special_tokens=False)
                print(f"ID токена: {token_id}, Токен: '{decoded_token}'")

            completion = self._tokenizer.decode(completion_tokens, skip_special_tokens=True)

            return CompletionResponse(text=completion, raw={"model_output": tokens})

        @override
        def _tokenizer_messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
            messages_dict = []
            for message in messages:
                if message.role == MessageRole.SYSTEM:
                    messages_dict.append({"role": "system", "content": message.content})
                elif message.role == MessageRole.USER:
                    messages_dict.append({"role": "user", "content": message.content})
                elif message.role == MessageRole.ASSISTANT:
                    messages_dict.append({"role": "assistant", "content": message.content})

            try:
                return self._tokenizer.apply_chat_template(
                    messages_dict,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except Exception as e:
                prompt = ""
                for msg in messages_dict:
                    if msg["role"] == "system":
                        prompt += f"<|system|>\n{msg['content']}</s>\n"
                    elif msg["role"] == "user":
                        prompt += f"<|user|>\n{msg['content']}</s>\n"
                    elif msg["role"] == "assistant":
                        prompt += f"<|assistant|>\n{msg['content']}</s>\n"
                prompt += "<|assistant|>\n"
                return prompt


    class FridaEmbeddings(BaseEmbedding):
        _model: SentenceTransformer = PrivateAttr()

        def __init__(
                self,
                model_name: str = "ai-forever/FRIDA",
                **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self._model = SentenceTransformer(model_name)

        async def _aget_query_embedding(self, query: str) -> List[float]:
            return self._get_query_embedding(query)

        async def _aget_text_embedding(self, text: str) -> List[float]:
            return self._get_text_embedding(text)

        def _get_query_embedding(self, query: str) -> List[float]:
            embeddings = self._model.encode([query])
            return embeddings[0].tolist()

        def _get_text_embedding(self, text: str) -> List[float]:
            embeddings = self._model.encode([text])
            return embeddings[0].tolist()

        def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
            embeddings = self._model.encode(texts)
            return embeddings.tolist()