"""anchor и positive беру из дубля, негатив домываю от узкого бакета к широкому: свои же
кандидаты той же базы, потом param2, param1, подкатегория, категория. случайный негатив
отделяется тривиально, и башня упирается в потолок.
"""

from __future__ import annotations

import numpy as np

from .device import pick_device
from .models import _build_module

__all__ = ["SideRefView", "mine_triplets", "TripletTower", "DEFAULT_MARGIN", "LEVELS"]

DEFAULT_MARGIN = 0.3

# от узкого бакета к широкому
LEVELS = ("та же база", "param2", "param1", "подкатегория", "категория", "случайный")


class SideRefView:
    # сторона пары адресуется числом: 2*строка — база, 2*строка+1 — кандидат.
    # плотную матрицу не разворачиваю, собираю на батч
    def __init__(self, embeddings, bidx, cidx, num_base, num_cand) -> None:
        self.embeddings = embeddings
        self.bidx = np.asarray(bidx)
        self.cidx = np.asarray(cidx)
        self.num_base = np.asarray(num_base, dtype=np.float32)
        self.num_cand = np.asarray(num_cand, dtype=np.float32)
        self.width = embeddings.shape[1] + self.num_base.shape[1]

    def take(self, refs: np.ndarray) -> np.ndarray:
        refs = np.asarray(refs)
        rows = refs >> 1
        is_cand = (refs & 1).astype(bool)
        emb_idx = np.where(is_cand, self.cidx[rows], self.bidx[rows])
        emb = self.embeddings[emb_idx].astype(np.float32, copy=False)
        num = np.where(is_cand[:, None], self.num_cand[rows], self.num_base[rows])
        return np.hstack([emb, num])


def _bucket_pick(pool: np.ndarray, codes: np.ndarray, queries: np.ndarray, rng):
    # случайный элемент пула с тем же кодом, иначе -1
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    lo = np.searchsorted(sorted_codes, queries, side="left")
    hi = np.searchsorted(sorted_codes, queries, side="right")
    size = hi - lo
    out = np.full(len(queries), -1, dtype=np.int64)
    ok = size > 0
    if ok.any():
        offset = (rng.random(ok.sum()) * size[ok]).astype(np.int64)
        out[ok] = pool[order[lo[ok] + offset]]
    return out


def mine_triplets(
    base_id: np.ndarray,
    y: np.ndarray,
    side_keys: np.ndarray,
    rows: np.ndarray,
    *,
    n_negatives: int = 1,
    seed: int = 42,
    verbose: bool = True,
):
    # side_keys — коды атрибутов формы [2*строк, 4]: param2, param1, подкатегория, категория
    if n_negatives > 1:
        parts, merged = [], {}
        for k in range(n_negatives):
            a, p, n, stats = _mine_once(
                base_id, y, side_keys, rows, seed=seed + k, verbose=False
            )
            parts.append((a, p, n))
            for name, count in stats.items():
                merged[name] = merged.get(name, 0) + count
        anchors = np.concatenate([x[0] for x in parts])
        positives = np.concatenate([x[1] for x in parts])
        negatives = np.concatenate([x[2] for x in parts])
        if verbose:
            _report(len(anchors), merged)
        return anchors, positives, negatives, merged
    return _mine_once(base_id, y, side_keys, rows, seed=seed, verbose=verbose)


def _report(total: int, stats: dict) -> None:
    print(f"триплетов собрано: {total}", flush=True)
    for name, count in stats.items():
        if count:
            print(f"  негатив с уровня «{name}»: {count} ({count / total:.1%})", flush=True)


def _mine_once(
    base_id: np.ndarray,
    y: np.ndarray,
    side_keys: np.ndarray,
    rows: np.ndarray,
    *,
    seed: int = 42,
    verbose: bool = True,
):
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows)
    y_rows = y[rows]

    positive_rows = rows[y_rows == 1]
    negative_rows = rows[y_rows == 0]
    if len(positive_rows) == 0 or len(negative_rows) == 0:
        raise ValueError("для триплетов нужны и дубли, и не дубли")

    anchors = (positive_rows * 2).astype(np.int64)
    positives = anchors + 1
    negatives = np.full(len(anchors), -1, dtype=np.int64)

    # пул негативов — кандидаты из пар с меткой 0. они не дубли своей базы, и для чужого
    # якоря почти наверняка тоже
    pool = (negative_rows * 2 + 1).astype(np.int64)
    stats = {}

    # уровень 0: свои же кандидаты той же базы. самый тяжёлый негатив, какой есть в данных
    todo = negatives < 0
    picked = _bucket_pick(pool, base_id[negative_rows], base_id[positive_rows][todo], rng)
    negatives[todo] = picked
    stats[LEVELS[0]] = int((negatives >= 0).sum())

    # уровни 1..4: тот же param2, param1, подкатегория, категория
    for level in range(4):
        todo = negatives < 0
        if not todo.any():
            stats[LEVELS[level + 1]] = 0
            continue
        picked = _bucket_pick(
            pool, side_keys[pool, level], side_keys[anchors[todo], level], rng
        )
        negatives[todo] = picked
        stats[LEVELS[level + 1]] = int((negatives >= 0).sum() - sum(stats.values()))

    # остаток — случайный негатив
    todo = negatives < 0
    stats[LEVELS[5]] = int(todo.sum())
    if todo.any():
        negatives[todo] = rng.choice(pool, size=int(todo.sum()), replace=True)

    # негатив не может совпасть с якорем или позитивом
    bad = (negatives == anchors) | (negatives == positives)
    if bad.any():
        negatives[bad] = rng.choice(pool, size=int(bad.sum()), replace=True)

    if verbose:
        _report(len(anchors), stats)

    return anchors, positives, negatives, stats


class TripletTower:
    def __init__(
        self,
        input_dim: int,
        hidden: int = 256,
        output: int = 128,
        margin: float = DEFAULT_MARGIN,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 4096,
        epochs: int = 8,
        seed: int = 42,
        device: str | None = None,
        verbose: bool = True,
    ) -> None:
        self.input_dim = input_dim
        self.hidden = hidden
        self.output = output
        self.margin = margin
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.verbose = verbose
        self.device = device or pick_device(verbose=verbose)
        self.model = None

    def fit(self, view: SideRefView, triplets) -> "TripletTower":
        import torch
        from torch.nn import TripletMarginLoss

        torch.manual_seed(self.seed)
        anchors, positives, negatives = triplets

        self.model = _build_module(self.input_dim, self.hidden, self.output, self.dropout)
        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = TripletMarginLoss(margin=self.margin)

        rng = np.random.default_rng(self.seed)
        n = len(anchors)
        steps = max(n // self.batch_size, 1)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            order = rng.permutation(n)
            total = 0.0
            for step in range(steps):
                rows = order[step * self.batch_size : (step + 1) * self.batch_size]
                a = torch.from_numpy(view.take(anchors[rows])).to(self.device)
                p = torch.from_numpy(view.take(positives[rows])).to(self.device)
                ng = torch.from_numpy(view.take(negatives[rows])).to(self.device)

                optimizer.zero_grad()
                loss = loss_fn(
                    self.model.encode(a), self.model.encode(p), self.model.encode(ng)
                )
                loss.backward()
                optimizer.step()
                total += loss.item()
            if self.verbose:
                print(f"  эпоха {epoch}: loss {total / steps:.4f}", flush=True)
        return self

    def predict(self, base_view, cand_view, batch_size: int = 16384) -> np.ndarray:
        # тот же скор, что у контрастной башни, чтобы его можно было подставить вместо неё
        import torch

        if self.model is None:
            raise RuntimeError("модель не обучена")
        self.model.eval()
        n = len(base_view)
        out = np.empty(n, dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, n, batch_size):
                rows = np.arange(start, min(start + batch_size, n))
                b = torch.from_numpy(base_view.take(rows)).to(self.device)
                c = torch.from_numpy(cand_view.take(rows)).to(self.device)
                distance = self.model(b, c).cpu().numpy()
                out[rows] = 1.0 / (1.0 + distance)
        return out
