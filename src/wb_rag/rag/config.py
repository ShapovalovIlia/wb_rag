from dataclasses import dataclass


@dataclass
class Config:
    data_dir: str = "./data"
    persist_dir: str = "./chroma_data"
    collection: str = "kb1"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 200  # ~слов (грубая аппроксимация токенов)
    chunk_overlap: int = 40  # ~слов
    min_doc_chars: int = 200  # пропуск крошечных фрагментов
    top_k_dense: int = 5  # кандидатов из dense
    top_k_final: int = 3  # финальный контекст
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    language_tag: str | None = None  # проставляется в метаданные, если нужно
