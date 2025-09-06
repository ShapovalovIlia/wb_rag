from datetime import datetime
from pathlib import Path

from rich import print as rprint
import chromadb
from sentence_transformers import SentenceTransformer

from wb_rag.rag.config import Config
from wb_rag.rag.utils import chunk_text
from wb_rag.rag.utils import normalize_text, sha1


def discover_documents(data_dir: str) -> list[Path]:
    exts = {".txt", ".md", ".pdf"}
    files: list[Path] = []
    for ext in exts:
        files.extend(Path(data_dir).rglob(f"*{ext}"))
    return files


def load_document(path: Path) -> str:
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    file_type = path.suffix
    error_message = f"Unsupported file type: {file_type}"
    raise ValueError(error_message)


def ensure_collection(
    client: chromadb.ClientAPI, cfg: Config
) -> chromadb.Collection:
    col = client.get_or_create_collection(
        name=cfg.collection,
        metadata={"model": cfg.embed_model, "hnsw:space": "cosine"},
    )
    # Предостережение о смешивании моделей
    try:
        meta = col.metadata or {}
        existing = meta.get("model")
        if existing and existing != cfg.embed_model:
            rprint(
                f"[yellow]ВНИМАНИЕ:[/yellow] колл. '{cfg.collection}' создана с моделью '{existing}',\n"
                f"а сейчас выбрана '{cfg.embed_model}'. Создайте новую коллекцию или переиндексируйте."
            )
    except Exception:
        pass
    return col


def ingest(cfg: Config) -> None:
    client = chromadb.PersistentClient(path=cfg.persist_dir)
    col = ensure_collection(client, cfg)
    embed = SentenceTransformer(cfg.embed_model)

    files = discover_documents(cfg.data_dir)
    if not files:
        rprint(f"[red]Нет файлов для индексации в {cfg.data_dir}[/red]")
        return

    total_chunks = 0
    for path in files:
        try:
            raw = load_document(path)
        except Exception as e:
            rprint(f"[red]Ошибка загрузки {path}: {e}[/red]")
            continue
        text = normalize_text(raw)
        if len(text) < cfg.min_doc_chars:
            continue

        chunks = chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
        if not chunks:
            continue

        doc_id = sha1(str(path.resolve()))
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []
        for idx, (chunk, (start, end)) in enumerate(chunks):
            ids.append(f"{doc_id}:{idx}")
            docs.append(chunk)
            metas.append(
                {
                    "source": str(path),
                    "title": path.stem,
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "start_char": start,
                    "end_char": end,
                    "mtime": datetime.fromtimestamp(
                        path.stat().st_mtime
                    ).isoformat(),
                    "lang": cfg.language_tag or "unknown",
                }
            )

        # батчевое кодирование
        embs = embed.encode(
            docs, normalize_embeddings=True, batch_size=64
        ).tolist()
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        total_chunks += len(ids)
        rprint(f"[green]Индексирован[/green] {path} → {len(ids)} чанков")

    rprint(
        f"[bold green]Готово.[/bold green] Всего добавлено чанков: {total_chunks}"
    )
