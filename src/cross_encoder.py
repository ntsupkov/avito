"""башни кодируют стороны независимо и встречаются только на расстоянии, значит сопоставить
слово базы со словом кандидата они не могут. для дублей это дорого: iphone 11 256 гб и
iphone 11 128 гб дают почти одинаковые векторы, а различает их одно число.

кросс-энкодер получает обе стороны одной последовательностью через [SEP], внимание работает
между ними. цена — представление больше не посчитать заранее на объявление, скор считается
на каждую пару. тут дообучается сам энкодер, а не проекция над ним.
"""

from __future__ import annotations

import time

import numpy as np

from .device import pick_device

__all__ = ["CrossEncoderRanker", "DEFAULT_MODEL"]

DEFAULT_MODEL = "cointegrated/rubert-tiny2"


class CrossEncoderRanker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 128,
        batch_size: int = 128,
        epochs: int = 2,
        lr: float = 5e-5,
        negative_ratio: float | None = 4.0,
        max_seconds: float | None = None,
        seed: int = 42,
        device: str | None = None,
        verbose: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.negative_ratio = negative_ratio
        self.max_seconds = max_seconds
        self.seed = seed
        self.verbose = verbose
        self.device = device or pick_device(verbose=verbose)
        self.model = None
        self.tokenizer = None
        self.history: dict = {}

    def _subsample(self, y: np.ndarray) -> np.ndarray:
        # все позитивы и часть негативов
        positives = np.flatnonzero(y == 1)
        negatives = np.flatnonzero(y == 0)
        if self.negative_ratio is None:
            return np.concatenate([positives, negatives])
        keep = min(len(negatives), int(len(positives) * self.negative_ratio))
        rng = np.random.default_rng(self.seed)
        chosen = rng.choice(negatives, size=keep, replace=False)
        return np.concatenate([positives, chosen])

    def fit(
        self, texts: list[str], idx_a: np.ndarray, idx_b: np.ndarray, y: np.ndarray
    ) -> "CrossEncoderRanker":
        import torch
        from torch.nn import BCEWithLogitsLoss
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device != "cuda":
            raise RuntimeError(
                "кросс-энкодер требует рабочей видеокарты: на процессоре дообучение "
                "на миллионах пар не укладывается ни в какой разумный бюджет"
            )

        torch.manual_seed(self.seed)
        y = np.asarray(y, dtype=np.float32)
        index = self._subsample(y)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(index)

        # после прореживания доля позитивов выросла, значит вес класса считаю по тому,
        # что реально попало в обучение, а не по исходной выборке
        n_pos = float((y[index] == 1).sum())
        n_neg = float((y[index] == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=self.device)

        if self.verbose:
            print(
                f"  кросс-энкодер: {len(index)} пар из {len(y)} "
                f"(позитивов {n_pos:.0f}, негативов {n_neg:.0f}, вес {pos_weight.item():.2f})",
                flush=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=1
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)
        scaler = torch.amp.GradScaler("cuda")

        steps_per_epoch = max(len(index) // self.batch_size, 1)
        total_steps = steps_per_epoch * self.epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=0.1
        )

        started = time.time()
        stopped_early = False
        done_steps = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            rng.shuffle(index)
            running = 0.0
            for step in range(steps_per_epoch):
                rows = index[step * self.batch_size : (step + 1) * self.batch_size]
                batch = self.tokenizer(
                    [texts[i] for i in idx_a[rows]],
                    [texts[i] for i in idx_b[rows]],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                target = torch.from_numpy(y[rows]).to(self.device).unsqueeze(1)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = self.model(**batch).logits
                    loss = loss_fn(logits.float(), target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running += loss.item()
                done_steps += 1

                if self.max_seconds and time.time() - started > self.max_seconds:
                    stopped_early = True
                    break

            if self.verbose:
                print(
                    f"    эпоха {epoch}: loss {running / max(step + 1, 1):.4f}, "
                    f"{time.time() - started:.0f}s",
                    flush=True,
                )
            if stopped_early:
                break

        self.history = {
            "seconds": time.time() - started,
            "steps_done": done_steps,
            "steps_planned": total_steps,
            "stopped_early": stopped_early,
            "train_pairs": int(len(index)),
        }
        if stopped_early and self.verbose:
            print(
                f"  БЮДЖЕТ ИСЧЕРПАН: пройдено {done_steps} шагов из {total_steps} "
                f"({done_steps / total_steps:.0%})",
                flush=True,
            )
        return self

    def predict(
        self, texts: list[str], idx_a: np.ndarray, idx_b: np.ndarray, batch_size: int = 512
    ) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("модель не обучена")
        self.model.eval()
        out = np.empty(len(idx_a), dtype=np.float32)

        with torch.inference_mode():
            for start in range(0, len(idx_a), batch_size):
                end = min(start + batch_size, len(idx_a))
                batch = self.tokenizer(
                    [texts[i] for i in idx_a[start:end]],
                    [texts[i] for i in idx_b[start:end]],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = self.model(**batch).logits
                out[start:end] = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy()
        return out
