from __future__ import annotations
import os

import hashlib
from openai import OpenAI as OpenAIClient


def chunk_text(
    text: str, chunk_size: int, overlap: int
) -> list[tuple[str, tuple[int, int]]]:
    """Чанкинг по словам с overlap. Возвращает [(chunk_text, (start_char, end_char)), ...]."""
    words = text.split()
    if not words:
        return []
    chunks: list[tuple[str, tuple[int, int]]] = []
    step = max(1, chunk_size - overlap)
    i = 0
    # Для вычисления символов восстановим через join на лету
    while i < len(words):
        w_slice = words[i : i + chunk_size]
        chunk = " ".join(w_slice)
        # оценим позиции символов: грубо, но полезно для метаданных
        start_char = len(" ".join(words[:i]))
        end_char = start_char + len(chunk)
        chunks.append((chunk, (start_char, end_char)))
        i += step
    return chunks


# -------------------- LLM (OpenRouter) --------------------


def call_openrouter(
    prompt: str,
    model: str = "openai/gpt-oss-20b:free",
    base_url: str | None = None,
    temperature: float = 0.2,
) -> str:
    """Вызов LLM через OpenRouter (совместим с openai SDK).
    Требует переменную окружения OPENROUTER_API_KEY. Базовый URL можно задать через
    OPENROUTER_BASE_URL или аргумент base_url (по умолчанию https://openrouter.ai/api/v1).
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        error_message = "Установите переменную окружения OPENROUTER_API_KEY с вашим ключом OpenRouter"
        raise RuntimeError(error_message)
    base = base_url or os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    client = OpenAIClient(base_url=base, api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content


# -------------------- Промпт‑сборщик --------------------
def build_prompt(query: str, passages: list[dict]) -> str:
    header = (
        "Ты — строгий помощник. Отвечай коротко и точно, основываясь ТОЛЬКО на контексте.\n"
        "Если факта нет в контексте — так и скажи. В конце укажи источники в виде [S1], [S2].\n"
    )
    blocks = []
    citations = []
    for i, p in enumerate(passages, start=1):
        meta = p.get("meta", {})
        src = meta.get("source", "?")
        title = meta.get("title", "?")
        blocks.append(f"[S{i}] {title} — {src}\n{p['text']}")
        citations.append(f"[S{i}] {title}")
    context = "\n\n".join(blocks)

    prompt = f"{header}\nКОНТЕКСТ:\n{context}\n\nВОПРОС: {query}\nОТВЕТ:"
    return prompt


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def normalize_text(s: str) -> str:
    # упрощённая нормализация: пробелы, скрытые символы
    return " ".join(s.replace("\u00a0", " ").split())
