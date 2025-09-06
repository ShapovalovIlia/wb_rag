import argparse
import json
from rich import print as rprint
from rich.table import Table

from wb_rag.rag.config import Config
from wb_rag.rag.ingest import ingest
from wb_rag.rag.utils import build_prompt, call_openrouter
from wb_rag.rag.retriever import HybridRetriever


def cmd_ingest(args: argparse.Namespace) -> None:
    cfg = Config(
        data_dir=args.data,
        persist_dir=args.persist,
        collection=args.collection,
        embed_model=args.embed_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        language_tag=args.lang,
    )
    ingest(cfg)


def cmd_query(args: argparse.Namespace) -> None:
    cfg = Config(
        persist_dir=args.persist,
        collection=args.collection,
        embed_model=args.embed_model,
        top_k_dense=args.topk_dense,
        top_k_final=args.topk,
        use_reranker=args.use_reranker,
        reranker_model=args.reranker_model,
    )
    retriever = HybridRetriever(cfg)
    where = None
    if args.filter:
        try:
            where = json.loads(args.filter)
        except Exception as e:
            rprint(f"[yellow]Игнорирую некорректный --filter: {e}[/yellow]")
    hits = retriever.retrieve(args.q, where=where, top_k=cfg.top_k_final)

    # Таблица результатов
    table = Table(title="Кандидаты")
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Источник")
    table.add_column("Заголовок")
    table.add_column("Фрагмент")
    for i, h in enumerate(hits, start=1):
        meta = h.get("meta", {})
        src = str(meta.get("source", "?"))
        title = str(meta.get("title", "?"))
        fragment = (
            (h["text"][:160] + "…") if len(h["text"]) > 160 else h["text"]
        )
        table.add_row(str(i), src, title, fragment)
    rprint(table)

    # Сборка промпта
    prompt = build_prompt(args.q, hits)
    rprint("\n[bold]Промпт для LLM:[/bold]\n")
    rprint(prompt)

    # (опционально) сразу получить ответ от OpenRouter
    if args.openrouter:
        try:
            answer = call_openrouter(
                prompt,
                model=args.or_model,
                base_url=args.or_base_url,
                temperature=args.temperature,
            )
            rprint("\n[bold]Ответ LLM (OpenRouter):[/bold]\n")
            rprint(answer)
        except Exception as e:
            rprint(f"[red]Ошибка вызова OpenRouter:[/red] {e}")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG + Chroma: шаблон")
    sub = p.add_subparsers(required=True)

    # ingest
    pi = sub.add_parser("ingest", help="Индексировать файлы из каталога")
    pi.add_argument(
        "--data",
        default="../rag/data",
        help="Каталог с документами (.txt/.md/.pdf)",
    )
    pi.add_argument(
        "--persist",
        default="../rag/chroma_data",
        help="Каталог для данных Chroma",
    )
    pi.add_argument("--collection", default="kb1", help="Имя коллекции Chroma")
    pi.add_argument(
        "--embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Модель эмбеддингов",
    )
    pi.add_argument(
        "--chunk-size", type=int, default=800, help="Размер чанка (слова)"
    )
    pi.add_argument(
        "--chunk-overlap",
        type=int,
        default=120,
        help="Перекрытие чанков (слова)",
    )
    pi.add_argument(
        "--lang",
        default=None,
        help="Языковой тег в метаданные (например, 'ru' или 'en')",
    )
    pi.set_defaults(func=cmd_ingest)

    # query
    pq = sub.add_parser(
        "query",
        help="Выполнить поиск и собрать промпт (+опц. получить ответ от LLM)",
    )
    pq.add_argument("--q", required=True, help="Пользовательский вопрос")
    pq.add_argument(
        "--persist", default="./chroma_data", help="Каталог с данными Chroma"
    )
    pq.add_argument("--collection", default="kb1", help="Имя коллекции Chroma")
    pq.add_argument(
        "--embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Модель эмбеддингов",
    )
    pq.add_argument(
        "--topk-dense",
        type=int,
        default=20,
        help="Dense кандидатов (до слияния)",
    )
    pq.add_argument(
        "--topk", type=int, default=6, help="Финальный топ-k для контекста"
    )
    pq.add_argument(
        "--filter", default=None, help="JSON-фильтр метаданных Chroma (where)"
    )
    pq.add_argument(
        "--use-reranker",
        action="store_true",
        help="Включить кросс-энкодерный реранк",
    )
    pq.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Модель реранкера",
    )

    # OpenRouter (LLM)
    pq.add_argument(
        "--openrouter",
        action="store_true",
        help="Отправить промпт в OpenRouter и вывести ответ",
    )
    pq.add_argument(
        "--or-model",
        default="openai/gpt-oss-20b:free",
        help="Имя модели в OpenRouter",
    )
    pq.add_argument(
        "--or-base-url",
        default=None,
        help="Базовый URL OpenRouter (по умолчанию https://openrouter.ai/api/v1)",
    )
    pq.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Температура генерации LLM",
    )

    pq.set_defaults(func=cmd_query)

    return p


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)
