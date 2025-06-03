from __future__ import annotations

import numpy as np

__all__ = ["rank_normalize", "rank_blend"]


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([0.5])

    order = np.argsort(scores, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)

    # одинаковым значениям — средний ранг, иначе порядок внутри ties зависел бы от порядка строк
    sorted_scores = scores[order]
    starts = np.flatnonzero(np.concatenate(([True], sorted_scores[1:] != sorted_scores[:-1])))
    sizes = np.diff(np.concatenate((starts, [n])))
    group_mean = np.repeat(
        (starts + (starts + sizes - 1)) / 2.0, sizes
    )
    ranks[order] = group_mean

    return ranks / (n - 1)


def rank_blend(scores_a: np.ndarray, scores_b: np.ndarray, weight: float = 0.5) -> np.ndarray:
    # weight — вес первого набора: 0.0 только второй, 1.0 только первый
    if not 0.0 <= weight <= 1.0:
        raise ValueError("вес должен лежать в [0, 1]")
    return weight * rank_normalize(scores_a) + (1.0 - weight) * rank_normalize(scores_b)
