import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from rich import print as rprint

from wb_rag.rag.config import Config
from wb_rag.rag.ingest import ensure_collection


class HybridRetriever:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = chromadb.PersistentClient(path=cfg.persist_dir)
        self.col = ensure_collection(self.client, cfg)
        self.embed = SentenceTransformer(cfg.embed_model)
        # для BM25 соберём корпус из коллекции (один раз на запуск)
        self._bm25_ids: list[str] = []
        self._bm25_docs: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._build_bm25()
        self._reranker = None
        if cfg.use_reranker and CrossEncoder is not None:
            try:
                self._reranker = CrossEncoder(cfg.reranker_model)
            except Exception as e:
                rprint(
                    f"[yellow]Не удалось инициализировать реранкер: {e}[/yellow]"
                )

    def _build_bm25(self) -> None:
        # Вычитаем все документы коллекции батчами (простая реализация)
        # Chroma не имеет прямого list_all, обойдёмся через peek
        try:
            peek = self.col.peek()  # может вернуть ограниченный набор
            # Если корпус маленький — достаточно, иначе пользователь может расширить
        except Exception:
            peek = None
        # Попытаемся пройтись сканированием через начальный id (best effort)
        # Для простоты используем peek
        if peek and "documents" in peek:
            docs = peek["documents"]
            ids = peek["ids"]
            self._bm25_ids = ids
            self._bm25_docs = [d.split() for d in docs]
            if self._bm25_docs:
                self._bm25 = BM25Okapi(self._bm25_docs)
        else:
            self._bm25 = None

    @staticmethod
    def _rrf(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    def retrieve(
        self,
        query: str,
        where: dict | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        top_k = top_k or self.cfg.top_k_final
        q_emb = self.embed.encode([query], normalize_embeddings=True)[
            0
        ].tolist()

        dense = self.col.query(
            query_embeddings=[q_emb],
            n_results=max(self.cfg.top_k_dense, top_k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        dense_ids = dense["ids"][0]
        dense_docs = dense["documents"][0]
        dense_meta = dense["metadatas"][0]
        # в Chroma при cosine меньшая distance — лучше; порядок уже по релевантности
        combined: dict[str, float] = {}
        for rank, did in enumerate(dense_ids):
            combined[did] = combined.get(did, 0.0) + self._rrf(rank)

        if self._bm25 is not None and self._bm25_docs:
            q_tokens = query.split()
            scores = self._bm25.get_scores(q_tokens)
            # возьмём top_s по BM25
            top_s = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[: max(50, top_k)]
            for r, idx in enumerate(top_s):
                did = self._bm25_ids[idx]
                combined[did] = combined.get(did, 0.0) + self._rrf(r)

        # Соберём кандидатов: часть из dense + часть из bm25 корпуса (при наличии)
        # Для текстов/метаданных тех, кого нет в dense, дотянем через where_document заглушкой
        candidates: dict[str, dict] = {}
        for did, doc, meta in zip(dense_ids, dense_docs, dense_meta):
            candidates[did] = {"id": did, "text": doc, "meta": meta}

        # Если BM25 добавил новые id, попробуем запросить их содержимое по ids через get()
        extra_ids = [did for did in combined.keys() if did not in candidates]
        if extra_ids:
            try:
                got = self.col.get(
                    ids=extra_ids, include=["documents", "metadatas", "ids"]
                )
                for did, doc, meta in zip(
                    got.get("ids", []),
                    got.get("documents", []),
                    got.get("metadatas", []),
                ):
                    candidates[did] = {"id": did, "text": doc, "meta": meta}
            except Exception:
                pass

        ranked = sorted(
            candidates.values(),
            key=lambda x: combined.get(x["id"], 0.0),
            reverse=True,
        )
        ranked = ranked[: max(20, top_k)]  # оставим запас для опц. реранка

        # (опционально) кросс‑энкодерный реранк
        if self._reranker is not None and ranked:
            pairs = [(query, c["text"]) for c in ranked]
            try:
                scores = self._reranker.predict(pairs)
                for i, sc in enumerate(scores):
                    ranked[i]["ce_score"] = float(sc)
                ranked.sort(key=lambda x: x.get("ce_score", 0.0), reverse=True)
            except Exception as e:
                rprint(f"[yellow]Реранкер не сработал: {e}[/yellow]")

        return ranked[:top_k]
