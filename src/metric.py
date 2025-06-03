from __future__ import annotations

import numpy as np

__all__ = ["mean_average_precision", "average_precision", "map_by_group"]


def average_precision(relevance: np.ndarray) -> float:
    # relevance — 0/1 в порядке убывания скора
    rel = np.asarray(relevance, dtype=np.float64)
    n_pos = rel.sum()
    if n_pos == 0:
        return 0.0
    hits = np.cumsum(rel)
    ranks = np.arange(1, len(rel) + 1)
    return float((hits / ranks * rel).sum() / n_pos)


def map_by_group(
    groups: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    skip_empty: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # AP по всем группам за одну сортировку — цикл по группам на миллионах пар не живёт
    groups = np.asarray(groups)
    y = np.asarray(y_true).astype(np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if not (len(groups) == len(y) == len(s)):
        raise ValueError("groups, y_true и scores должны быть одной длины")
    if len(groups) == 0:
        empty = np.array([])
        return empty, empty, empty

    # первичный ключ — группа, вторичный — скор по убыванию; в lexsort главный ключ последний
    order = np.lexsort((-s, groups))
    g = groups[order]
    y = y[order]

    n = len(y)
    starts = np.flatnonzero(np.concatenate(([True], g[1:] != g[:-1])))
    sizes = np.diff(np.concatenate((starts, [n])))
    row_start = np.repeat(starts, sizes)  # начало своей группы для каждой строки
    rank = np.arange(n) - row_start + 1  # позиция внутри группы, с 1

    # cum_excl[i] = сумма y[:i], значит попаданий до i включительно = cum_excl[i+1] - cum_excl[start]
    cum_excl = np.concatenate(([0.0], np.cumsum(y)))
    hits = cum_excl[1:] - cum_excl[row_start]

    contrib = hits / rank * y
    ap_sum = np.add.reduceat(contrib, starts)
    n_pos = np.add.reduceat(y, starts)
    ap = np.divide(ap_sum, n_pos, out=np.zeros_like(ap_sum), where=n_pos > 0)

    group_ids = g[starts]
    if skip_empty:
        # базы без единого дубля не в счёт, для них AP не определён
        keep = n_pos > 0
        return group_ids[keep], ap[keep], n_pos[keep]
    return group_ids, ap, n_pos


def mean_average_precision(
    groups: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    skip_empty: bool = True,
) -> float:
    # среднее AP по базам. на сплошных ties зависит от порядка строк, для константы бессмысленна
    _, ap, _ = map_by_group(groups, y_true, scores, skip_empty=skip_empty)
    return float(ap.mean()) if len(ap) else 0.0
