"""признаки описывают пару, а не объявление: модель видит только сходство base и cand."""

from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd

from .text import jaccard, normalize, tokenize

__all__ = [
    "build_features",
    "TEXT_PAIRS",
    "add_embedding_features",
    "embedding_pair_features",
    "side_numeric_features",
]

# поле и размер символьной n-граммы. у заголовков n=3, они короткие и длиннее просто
# не наберётся; у описаний n=4 — текста хватает, а случайных совпадений меньше
TEXT_PAIRS: tuple[tuple[str, int], ...] = (("title", 3), ("description", 4))

MAX_DESCRIPTION_CHARS = 512


def _safe_str(series: pd.Series) -> np.ndarray:
    return series.fillna("").astype(str).to_numpy()


def _containment(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _lcp_ratio(a: str, b: str) -> float:
    # общий префикс — ловит iphone 11 128 гб против iphone 11 256 гб
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / max(len(a), len(b))


def _ngrams(text: str, n: int) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


class _BoundedCache:
    """пары одной базы идут подряд, поэтому кэш разобранных текстов окупается.
    но только ограниченный: без потолка он разросся до 12.6 гб на одном файле.
    """

    __slots__ = ("_prepare", "_maxsize", "_data")

    def __init__(self, prepare, maxsize: int = 4096) -> None:
        self._prepare = prepare
        self._maxsize = maxsize
        self._data: dict = {}

    def get(self, raw: str):
        hit = self._data.get(raw)
        if hit is None:
            hit = self._prepare(raw)
            if len(self._data) >= self._maxsize:
                self._data.clear()
            self._data[raw] = hit
        return hit


def _text_similarity_block(
    base_raw: np.ndarray, cand_raw: np.ndarray, field: str, ngram: int, max_chars: int | None
) -> dict[str, np.ndarray]:
    n = len(base_raw)
    out = {
        f"{field}_exact": np.zeros(n, dtype=np.float32),
        f"{field}_jaccard_tok": np.zeros(n, dtype=np.float32),
        f"{field}_containment_tok": np.zeros(n, dtype=np.float32),
        f"{field}_jaccard_ngram": np.zeros(n, dtype=np.float32),
        f"{field}_digits_jaccard": np.zeros(n, dtype=np.float32),
        f"{field}_lcp_ratio": np.zeros(n, dtype=np.float32),
        f"{field}_len_ratio": np.zeros(n, dtype=np.float32),
        f"{field}_len_diff": np.zeros(n, dtype=np.float32),
        f"{field}_both_empty": np.zeros(n, dtype=np.float32),
    }
    def prepare(raw: str):
        norm = normalize(raw[:max_chars] if max_chars else raw)
        toks = norm.split()
        return norm, set(toks), _ngrams(norm, ngram), {t for t in toks if t.isdigit()}

    cache = _BoundedCache(prepare)

    for i in range(n):
        b_norm, b_tok, b_ng, b_dig = cache.get(base_raw[i])
        c_norm, c_tok, c_ng, c_dig = cache.get(cand_raw[i])

        if not b_norm and not c_norm:
            out[f"{field}_both_empty"][i] = 1.0
            continue

        out[f"{field}_exact"][i] = float(b_norm == c_norm and bool(b_norm))
        out[f"{field}_jaccard_tok"][i] = jaccard(b_tok, c_tok)
        out[f"{field}_containment_tok"][i] = _containment(b_tok, c_tok)
        out[f"{field}_jaccard_ngram"][i] = jaccard(b_ng, c_ng)
        if b_dig or c_dig:
            out[f"{field}_digits_jaccard"][i] = jaccard(b_dig, c_dig)
        else:
            out[f"{field}_digits_jaccard"][i] = -1.0  # чисел нет ни там, ни там
        out[f"{field}_lcp_ratio"][i] = _lcp_ratio(b_norm, c_norm)

        lb, lc = len(b_norm), len(c_norm)
        out[f"{field}_len_ratio"][i] = min(lb, lc) / max(lb, lc) if max(lb, lc) else 0.0
        out[f"{field}_len_diff"][i] = abs(lb - lc)

    return out


def _parse_params(raw: str) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_params_block(base_raw: np.ndarray, cand_raw: np.ndarray) -> dict[str, np.ndarray]:
    n = len(base_raw)
    out = {
        "params_n_common_keys": np.zeros(n, dtype=np.float32),
        "params_key_jaccard": np.zeros(n, dtype=np.float32),
        "params_value_match_ratio": np.zeros(n, dtype=np.float32),
        "params_n_base": np.zeros(n, dtype=np.float32),
        "params_n_cand": np.zeros(n, dtype=np.float32),
    }
    cache = _BoundedCache(_parse_params)

    for i in range(n):
        b, c = cache.get(base_raw[i]), cache.get(cand_raw[i])
        out["params_n_base"][i] = len(b)
        out["params_n_cand"][i] = len(c)
        if not b or not c:
            out["params_value_match_ratio"][i] = -1.0
            continue
        bk, ck = set(b), set(c)
        common = bk & ck
        out["params_n_common_keys"][i] = len(common)
        out["params_key_jaccard"][i] = len(common) / len(bk | ck)
        if common:
            equal = sum(1 for k in common if b[k] == c[k])
            out["params_value_match_ratio"][i] = equal / len(common)
        else:
            out["params_value_match_ratio"][i] = -1.0

    return out


def _clean_price(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    return np.where(values > 0, values, np.nan)


def side_numeric_features(df: pd.DataFrame, prefix: str) -> np.ndarray:
    price = _clean_price(df[f"{prefix}_price"])
    log_price = np.log1p(np.nan_to_num(price, nan=0.0))
    price_known = np.isfinite(price).astype(np.float64)

    images = pd.to_numeric(df[f"{prefix}_count_images"], errors="coerce").fillna(0).to_numpy()
    title_len = df[f"{prefix}_title"].fillna("").str.len().to_numpy()
    desc_len = df[f"{prefix}_description"].fillna("").str.len().to_numpy()

    out = np.column_stack([
        log_price, price_known, images, np.log1p(title_len), np.log1p(desc_len)
    ]).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError(f"в признаках стороны {prefix} остались NaN или бесконечности")
    return out


def _numeric_block(df: pd.DataFrame) -> dict[str, np.ndarray]:
    # цена, число фотографий и готовые гео-флаги
    out: dict[str, np.ndarray] = {}

    bp = _clean_price(df["base_price"])
    cp = _clean_price(df["cand_price"])
    lo, hi = np.minimum(bp, cp), np.maximum(bp, cp)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["price_ratio"] = np.where(hi > 0, lo / hi, np.nan)
        out["price_log_diff"] = np.abs(np.log1p(bp) - np.log1p(cp))
    out["price_abs_diff"] = np.abs(bp - cp)
    out["price_min"] = lo
    out["price_max"] = hi
    out["price_missing"] = (np.isnan(bp) | np.isnan(cp)).astype(np.float32)

    bi = pd.to_numeric(df["base_count_images"], errors="coerce").fillna(0).to_numpy()
    ci = pd.to_numeric(df["cand_count_images"], errors="coerce").fillna(0).to_numpy()
    out["images_abs_diff"] = np.abs(bi - ci)
    out["images_min"] = np.minimum(bi, ci)
    out["images_max"] = np.maximum(bi, ci)
    out["images_equal"] = (bi == ci).astype(np.float32)

    for col in ("is_same_location", "is_same_region"):
        if col in df.columns:
            out[col] = df[col].fillna(False).astype(np.float32).to_numpy()

    return out


def _categorical_block(df: pd.DataFrame) -> dict[str, np.ndarray]:
    # разные категории почти исключают дубль, то есть это работает фильтром, а не слабым сигналом
    out: dict[str, np.ndarray] = {}
    for field in ("category_name", "subcategory_name", "param1", "param2"):
        b_col, c_col = f"base_{field}", f"cand_{field}"
        if b_col not in df.columns or c_col not in df.columns:
            continue
        b, c = _safe_str(df[b_col]), _safe_str(df[c_col])
        out[f"same_{field}"] = (b == c).astype(np.float32)
        out[f"{field}_missing"] = ((b == "") | (c == "")).astype(np.float32)
    return out


def _title_image_block(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if "base_title_image" not in df.columns:
        return out
    b = _safe_str(df["base_title_image"])
    c = _safe_str(df["cand_title_image"])
    out["title_image_both_present"] = ((b != "") & (c != "")).astype(np.float32)
    out["title_image_equal"] = (b == c).astype(np.float32)
    return out


def _cross_block(df: pd.DataFrame) -> dict[str, np.ndarray]:
    # перепост часто копирует заголовок в описание, значит такое вложение ловит дубли,
    # у которых сами заголовки переписаны и напрямую не совпадают
    n = len(df)
    titles = {p: _safe_str(df[f"{p}_title"]) for p in ("base", "cand")}
    descriptions = {p: _safe_str(df[f"{p}_description"]) for p in ("base", "cand")}

    title_cache = _BoundedCache(lambda raw: set(tokenize(raw)))
    desc_cache = _BoundedCache(lambda raw: set(tokenize(raw[:MAX_DESCRIPTION_CHARS])))

    out = {
        "cross_base_title_in_cand_desc": np.zeros(n, dtype=np.float32),
        "cross_cand_title_in_base_desc": np.zeros(n, dtype=np.float32),
    }
    for i in range(n):
        out["cross_base_title_in_cand_desc"][i] = _containment(
            title_cache.get(titles["base"][i]), desc_cache.get(descriptions["cand"][i])
        )
        out["cross_cand_title_in_base_desc"][i] = _containment(
            title_cache.get(titles["cand"][i]), desc_cache.get(descriptions["base"][i])
        )
    return out


def build_features(df: pd.DataFrame, *, verbose: bool = False) -> pd.DataFrame:
    blocks: dict[str, np.ndarray] = {}

    for field, ngram in TEXT_PAIRS:
        if verbose:
            print(f"  признаки по полю {field}...", flush=True)
        max_chars = MAX_DESCRIPTION_CHARS if field == "description" else None
        blocks.update(
            _text_similarity_block(
                _safe_str(df[f"base_{field}"]),
                _safe_str(df[f"cand_{field}"]),
                field,
                ngram,
                max_chars,
            )
        )

    if "base_json_params" in df.columns:
        if verbose:
            print("  признаки по json_params...", flush=True)
        blocks.update(
            _json_params_block(_safe_str(df["base_json_params"]), _safe_str(df["cand_json_params"]))
        )

    if verbose:
        print("  числовые и категориальные...", flush=True)
    blocks.update(_numeric_block(df))
    blocks.update(_categorical_block(df))
    blocks.update(_title_image_block(df))

    if verbose:
        print("  перекрёстные заголовок/описание...", flush=True)
    blocks.update(_cross_block(df))

    out = pd.DataFrame(blocks, index=df.index)
    return out.astype(np.float32)


def add_embedding_features(
    features: pd.DataFrame,
    base_emb: np.ndarray,
    cand_emb: np.ndarray,
    prefix: str = "emb",
) -> pd.DataFrame:
    b = np.ascontiguousarray(base_emb, dtype=np.float32)
    c = np.ascontiguousarray(cand_emb, dtype=np.float32)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    cn = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-8)

    diff = np.abs(bn - cn)
    features = features.copy()
    features[f"{prefix}_cosine"] = (bn * cn).sum(axis=1).astype(np.float32)
    features[f"{prefix}_l2"] = np.linalg.norm(bn - cn, axis=1).astype(np.float32)
    features[f"{prefix}_diff_mean"] = diff.mean(axis=1).astype(np.float32)
    features[f"{prefix}_diff_max"] = diff.max(axis=1).astype(np.float32)
    features[f"{prefix}_diff_std"] = diff.std(axis=1).astype(np.float32)
    return features


def embedding_pair_features(
    embeddings: np.ndarray,
    base_idx: np.ndarray,
    cand_idx: np.ndarray,
    prefix: str = "emb",
    chunk: int = 200_000,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    n = len(base_idx)
    cols = ["cosine", "l2", "diff_mean", "diff_max", "diff_std"]
    out = {f"{prefix}_{c}": np.empty(n, dtype=np.float32) for c in cols}

    def take(idx: np.ndarray) -> np.ndarray:
        rows = embeddings[idx].astype(np.float32, copy=False)
        norms = np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-8)
        return rows / norms

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        b = take(base_idx[start:end])
        c = take(cand_idx[start:end])
        diff = np.abs(b - c)
        out[f"{prefix}_cosine"][start:end] = (b * c).sum(axis=1)
        out[f"{prefix}_l2"][start:end] = np.linalg.norm(b - c, axis=1)
        out[f"{prefix}_diff_mean"][start:end] = diff.mean(axis=1)
        out[f"{prefix}_diff_max"][start:end] = diff.max(axis=1)
        out[f"{prefix}_diff_std"][start:end] = diff.std(axis=1)

    return pd.DataFrame(out, index=index)
