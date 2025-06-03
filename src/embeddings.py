"""эмбеддинги rubert-tiny2."""

from __future__ import annotations

import numpy as np

from .device import pick_device
from .text import restore_homoglyphs

__all__ = ["TextIndex", "encode_texts", "DEFAULT_MODEL"]

DEFAULT_MODEL = "cointegrated/rubert-tiny2"


class TextIndex:
    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self.texts: list[str] = []

    def add(self, text: str) -> int:
        idx = self._ids.get(text)
        if idx is None:
            idx = len(self.texts)
            self._ids[text] = idx
            self.texts.append(text)
        return idx

    def add_many(self, texts) -> np.ndarray:
        return np.fromiter((self.add(t) for t in texts), dtype=np.int64, count=len(texts))

    def __len__(self) -> int:
        return len(self.texts)


def build_bert_text(title: str, description: str, desc_chars: int = 128) -> str:
    title = restore_homoglyphs(title or "")
    desc = restore_homoglyphs((description or "")[:desc_chars])
    return f"{title}. {desc}".strip()


def encode_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 512,
    max_length: int = 64,
    device: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = pick_device(verbose=verbose)
    if verbose:
        print(f"кодирую {len(texts)} уникальных текстов на {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    # float16: уникальных текстов миллионы, в float32 таблица занимает больше, чем все
    # признаки вместе. точность не нужна — дальше из векторов только косинус и статистики
    dim = model.config.hidden_size
    out = np.empty((len(texts), dim), dtype=np.float16)

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out[start : start + len(batch)] = pooled.float().cpu().numpy().astype(np.float16)
            if verbose and start and start % (batch_size * 200) == 0:
                print(f"  {start}/{len(texts)}", flush=True)

    return out
