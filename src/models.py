"""косинус сырых эмбеддингов бустингу почти ничего не даёт: модель не дообучается и меряет
смысловую близость, а два зимних пуховика близки по смыслу, но не дубли. мне нужна близость
в смысле дубля, значит её надо выучивать.

замороженный энкодер даёт представление, поверх учу проекцию — она стягивает дубли и разводит
недубли. обе стороны идут через один энкодер независимо и встречаются только на расстоянии
между проекциями, то есть модель не может подсмотреть в кандидата, кодируя базу.
"""

from __future__ import annotations

import numpy as np

from .device import pick_device

__all__ = ["TwoTowerContrastive", "contrastive_loss", "build_side_matrix", "SideView"]


class SideView:
    def __init__(self, embeddings: np.ndarray, idx: np.ndarray, numeric: np.ndarray) -> None:
        if len(idx) != len(numeric):
            raise ValueError("длина индексов и числовых признаков не совпадает")
        self.embeddings = embeddings
        self.idx = np.asarray(idx)
        self.numeric = np.asarray(numeric, dtype=np.float32)
        self.shape = (len(self.idx), embeddings.shape[1] + self.numeric.shape[1])

    def __len__(self) -> int:
        return self.shape[0]

    def take(self, rows: np.ndarray) -> np.ndarray:
        emb = self.embeddings[self.idx[rows]].astype(np.float32, copy=False)
        return np.hstack([emb, self.numeric[rows]])

    def subset(self, rows: np.ndarray) -> "SideView":
        # вид на подмножество — нарезка по фолдам без копирования
        return SideView(self.embeddings, self.idx[rows], self.numeric[rows])


class _DenseView:
    # тот же интерфейс поверх обычного массива, чтобы модель не различала случаи
    def __init__(self, array: np.ndarray) -> None:
        self.array = np.asarray(array, dtype=np.float32)
        self.shape = self.array.shape

    def __len__(self) -> int:
        return len(self.array)

    def take(self, rows: np.ndarray) -> np.ndarray:
        return self.array[rows]

    def subset(self, rows: np.ndarray) -> "_DenseView":
        return _DenseView(self.array[rows])


def _as_view(x):
    return x if hasattr(x, "take") and hasattr(x, "subset") else _DenseView(x)


def _torch():
    import torch

    return torch


def contrastive_loss(distance, label, margin: float = 1.0):
    # дубли штрафуются за расстояние, недубли — за то, что подошли ближе зазора. недубли
    # дальше margin вклада не дают, иначе обучение уходит в раздувание расстояний
    torch = _torch()
    positive = label * distance.pow(2)
    negative = (1.0 - label) * torch.clamp(margin - distance, min=0.0).pow(2)
    return (positive + negative).mean()


def _build_module(input_dim: int, hidden: int, output: int, dropout: float):
    torch = _torch()
    nn = torch.nn

    class TwoTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, output),
            )

        def encode(self, x):
            z = self.encoder(x)
            # l2: без неё расстояние можно уменьшать, просто сжимая представление,
            # и зазор перестаёт что-либо значить
            return z / z.norm(dim=1, keepdim=True).clamp(min=1e-8)

        def forward(self, x_base, x_cand):
            z_base = self.encode(x_base)
            z_cand = self.encode(x_cand)
            delta = z_base - z_cand
            return torch.sqrt((delta * delta).sum(dim=1) + 1e-12)

    return TwoTower()


class TwoTowerContrastive:
    def __init__(
        self,
        input_dim: int,
        hidden: int = 256,
        output: int = 128,
        margin: float = 1.0,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 4096,
        epochs: int = 5,
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

    def fit(self, x_base, x_cand, y: np.ndarray) -> "TwoTowerContrastive":
        torch = _torch()
        torch.manual_seed(self.seed)

        base_view, cand_view = _as_view(x_base), _as_view(x_cand)
        self.model = _build_module(self.input_dim, self.hidden, self.output, self.dropout)
        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        y = np.asarray(y, dtype=np.float32)
        n = len(y)
        # дублей ~5%: без выравнивания батч почти целиком из недублей, градиент от
        # положительной части тонет в шуме и модель сходится к «всё далеко»
        positive_idx = np.flatnonzero(y == 1)
        negative_idx = np.flatnonzero(y == 0)
        if len(positive_idx) == 0 or len(negative_idx) == 0:
            raise ValueError("в обучающей выборке должны быть оба класса")
        rng = np.random.default_rng(self.seed)
        half = max(self.batch_size // 2, 1)
        steps = max(n // self.batch_size, 1)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total = 0.0
            for _ in range(steps):
                rows = np.concatenate([
                    rng.choice(positive_idx, size=half, replace=len(positive_idx) < half),
                    rng.choice(negative_idx, size=half, replace=len(negative_idx) < half),
                ])
                xb = torch.from_numpy(base_view.take(rows)).to(self.device)
                xc = torch.from_numpy(cand_view.take(rows)).to(self.device)
                yt = torch.from_numpy(y[rows]).to(self.device)

                optimizer.zero_grad()
                loss = contrastive_loss(self.model(xb, xc), yt, self.margin)
                loss.backward()
                optimizer.step()
                total += loss.item()
            if self.verbose:
                print(f"  эпоха {epoch}: loss {total / steps:.4f}", flush=True)
        return self

    def predict(self, x_base, x_cand, batch_size: int = 16384) -> np.ndarray:
        # 1/(1+d), а не -d: так значения лежат в (0, 1] и годятся как признак наравне с остальными
        torch = _torch()
        if self.model is None:
            raise RuntimeError("модель не обучена")
        base_view, cand_view = _as_view(x_base), _as_view(x_cand)
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


def build_side_matrix(
    embeddings: np.ndarray, idx: np.ndarray, numeric: np.ndarray
) -> np.ndarray:
    # вход одной башни: эмбеддинг плюс числовые атрибуты. стандартизует вызывающий код
    emb = embeddings[idx].astype(np.float32, copy=False)
    return np.hstack([emb, numeric.astype(np.float32, copy=False)])
