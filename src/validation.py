"""одно объявление стоит и как base, и как cand, поэтому случайное разбиение по строкам
растаскивает его между train и valid и задирает оценку. режу по group_id, кластером целиком.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

__all__ = ["group_folds", "leakage_report"]


def group_folds(
    groups: np.ndarray, n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(groups)
    # GroupKFold раскладывает группы по размеру и не перемешивает, то есть сид на него
    # не влияет — перемешиваю сам, переименовав группы случайной перестановкой
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    shuffled = rng.permutation(len(uniq))
    remap = dict(zip(uniq, shuffled))
    permuted = np.array([remap[g] for g in groups])

    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups))
    return list(splitter.split(dummy, groups=permuted))


def leakage_report(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    id_cols: tuple[str, ...] = ("base_item_id", "cand_item_id"),
) -> dict[str, float]:
    # доля объявлений валидации, которые уже были в обучении. по group_id должна быть нулевой
    train_ids: set = set()
    valid_ids: set = set()
    for col in id_cols:
        train_ids |= set(df.iloc[train_idx][col].dropna())
        valid_ids |= set(df.iloc[valid_idx][col].dropna())
    if not valid_ids:
        return {"overlap_ratio": 0.0, "n_valid_items": 0, "n_leaked_items": 0}
    leaked = valid_ids & train_ids
    return {
        "overlap_ratio": len(leaked) / len(valid_ids),
        "n_valid_items": len(valid_ids),
        "n_leaked_items": len(leaked),
    }
