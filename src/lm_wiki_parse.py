from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.core import StorageContext, Settings, Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.callbacks import CallbackManager
from qdrant_client.http.models import VectorParams, Distance
from li_qdrant_handler import make_qdrant
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.readers.file import PDFReader
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

from qdrant_client import QdrantClient


def create_prompt(prompt: str) -> str:
    system_prompt = "Ты — эксперт по странам мира. Твоя задача — давать точные, структурированные и профессиональные ответы строго на основе предоставленного контекста из вики. Ответь кратко на вопрос пользователя, основываясь на контексте. Если данных недостаточно, напиши 'Недостаточно данных'. В конце обязательно напиши '|'!"
    return f"{system_prompt} {prompt}"


def initialize_qdrant_and_index():
    debug_handler = LlamaDebugHandler(print_trace_on_end=True)

    embed = HuggingFaceEmbedding(
        model_name="ai-forever/FRIDA",
        cache_folder="C://Users//thererealareyou//.cache//huggingface//hub",
    )

    reader = PDFReader()

    client = QdrantClient(path="./local_qdrant")

    # Проверяем существование коллекции перед созданием
    if not client.collection_exists("pages_chunks"):
        client.create_collection(
            collection_name="pages_chunks",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        print("Коллекция 'pages_chunks' создана")

        vector_store = QdrantVectorStore(
            client=client,
            collection_name="pages_chunks",
            embed_dimension=1536
        )

        splitter = TokenTextSplitter(
            chunk_size=512,
            chunk_overlap=100,
            separator=". ",
            backup_separators=["\n"]
        )

        nodes = splitter.get_nodes_from_documents(reader.load_data("./data/Russia.pdf"))

        index = VectorStoreIndex(
            nodes=nodes,
            vector_store=vector_store,
            embed_model=embed,
            show_progress=True
        )

        print(f"Добавлено {len(nodes)} чанков в новую коллекцию")
    else:
        print("Коллекция 'countries' уже существует")

        vector_store = QdrantVectorStore(
            client=client,
            collection_name="countries",
            embed_dimension=1536
        )

        # Проверяем количество записей в коллекции
        collection_info = client.get_collection("countries")

        client.close()

        print(f"Количество записей в существующей коллекции: {collection_info.points_count}")

        if collection_info.points_count == 0:
            make_qdrant()
        else:
            # Загружаем индекс из существующего векторного хранилища
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=embed,
                show_progress=True
            )
            print("Индекс загружен из существующей коллекции")

    print("Initializing Qwen and Frida...")

    llm = HuggingFaceLLM(
        model_name="Qwen/Qwen3-1.7B",
        tokenizer_name="Qwen/Qwen3-1.7B",
        device_map="cuda",
        completion_to_prompt=create_prompt,
        max_new_tokens=1,  # Увеличили количество токенов для более полного ответа
        tokenizer_kwargs={
            "enable_thinking": False,
        },
        generate_kwargs={
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.5,
            "pad_token_id": 151645,
            "early_stopping": True,
        }
    )

    Settings.llm = llm
    Settings.embed_model = embed
    Settings.callback_manager = CallbackManager([debug_handler])

    print("Creating query engine...")
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact"
    )

    return query_engine


print("Creating index and query engine...")
query_engine = initialize_qdrant_and_index()

print("Starting chat...")
while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    response = query_engine.query(user_input)
    print(response)

    print("\nИспользованные чанки:")
    for i, node in enumerate(response.source_nodes):
        print(f"--- Чанк {i + 1} (page_id: {node.node.metadata.get('page_id')}) ---")
        print(node.node.text)
        print()