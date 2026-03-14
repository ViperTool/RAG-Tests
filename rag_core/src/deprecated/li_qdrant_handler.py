import torch
from transformers import T5EncoderModel, AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.readers.file import PDFReader
import tqdm

def make_qdrant():
    PDF_PATH = "./data/Russia.pdf"

    COLLECTION_NAME = "pages_chunks"

    EMBEDDING_MODEL_NAME = "ai-forever/FRIDA"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CHUNK_SIZE = 256
    CHUNK_OVERLAP = 50

    print("Загрузка модели эмбеддингов...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    model = T5EncoderModel.from_pretrained(EMBEDDING_MODEL_NAME).to(DEVICE)
    model.eval()

    def get_embedding(text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            text = " "
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        return embeddings.tolist()

    print("Инициализация локального Qdrant (на диск)...")
    qdrant_client = QdrantClient(path="./local_qdrant")

    dummy_emb = get_embedding("test")
    vector_size = len(dummy_emb)
    print(vector_size)

    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Коллекция '{COLLECTION_NAME}' создана.")
    else:
        print(f"Коллекция '{COLLECTION_NAME}' уже существует.")

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator=" "
    )

    print("Загрузка и чанкирование данных из PDF...")
    reader = PDFReader()
    documents = reader.load_data(PDF_PATH)

    points = []
    point_id_counter = 0

    for doc_idx, doc in enumerate(tqdm.tqdm(documents, desc="Обработка документов")):
        if doc.text is None:
            continue
        if not isinstance(doc.text, str):
            doc.text = str(doc.text)
        if not doc.text.strip():
            continue

        doc.metadata = {"page_id": doc.metadata.get("page_label", f"pdf_{doc_idx}"), **doc.metadata}

        try:
            nodes = splitter.get_nodes_from_documents([doc])
        except Exception as e:
            print(f"Ошибка чанкирования для документа {doc_idx}: {e}")
            continue

        for node in nodes:
            chunk_text = node.text
            if chunk_text is None or not isinstance(chunk_text, str):
                chunk_text = " "

            emb = get_embedding(chunk_text)

            points.append(PointStruct(
                id=point_id_counter,
                vector=emb,
                payload={
                    "page_id": node.metadata.get("page_label", f"pdf_{doc_idx}") or f"pdf_{doc_idx}",
                    "text": chunk_text,
                    "chunk_index": node.node_id,
                    "pdf_document": PDF_PATH
                }
            ))
            point_id_counter += 1

            if len(points) >= 64:
                qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
                points.clear()

    if points:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Успешно загружено {point_id_counter} чанков в Qdrant из {PDF_PATH}.")