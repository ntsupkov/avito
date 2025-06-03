from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .device import pick_device

__all__ = [
    "ImageLocator",
    "ImageIndex",
    "DEFAULT_CLIP_MODEL",
    "dhash",
    "dhash_many",
    "hamming",
    "encode_images",
    "image_pair_features",
]

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")

DHASH_SIZE = 8

DRAFT_SIDE = 256


def _scandir_names(directory: Path, suffixes: tuple[str, ...]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    try:
        with os.scandir(directory) as it:
            for entry in it:
                stem, suffix = os.path.splitext(entry.name)
                if suffix.lower() in suffixes:
                    out.append((stem, suffix))
    except OSError:
        return []
    return out


class ImageLocator:
    def __init__(self, roots, *, suffixes=IMAGE_SUFFIXES, verbose: bool = False) -> None:
        self._roots = [Path(r) for r in roots if Path(r).exists()]
        self._suffixes = tuple(s.lower() for s in suffixes)
        self._verbose = verbose
        self._chunks: list[Path] = []
        self._suffix_of_chunk: list[str] = []
        self._position: np.ndarray | None = None
        self._keys: list[str] = []

    @classmethod
    def from_roots(cls, *roots, verbose: bool = False) -> "ImageLocator":
        return cls(roots, verbose=verbose)

    def _chunk_dirs(self) -> list[Path]:
        found: list[Path] = []
        for root in self._roots:
            found.append(root)
            try:
                with os.scandir(root) as it:
                    for entry in it:
                        if entry.is_dir():
                            found.append(Path(entry.path))
            except OSError:
                continue
        return found

    def build(self, keys, *, time_budget_s: float | None = None) -> "ImageLocator":
        import time

        self._keys = list(keys)
        wanted = {k: i for i, k in enumerate(self._keys) if k}
        self._position = np.full(len(self._keys), -1, dtype=np.int32)
        if not wanted:
            return self

        started = time.time()
        remaining = len(wanted)
        for chunk in self._chunk_dirs():
            if not remaining:
                break
            if time_budget_s is not None and time.time() - started > time_budget_s:
                if self._verbose:
                    print(
                        f"индекс фотографий: бюджет исчерпан, найдено "
                        f"{len(wanted) - remaining} из {len(wanted)}",
                        flush=True,
                    )
                break
            names = _scandir_names(chunk, self._suffixes)
            if not names:
                continue
            chunk_id = len(self._chunks)
            hits = 0
            suffix_seen = names[0][1]
            for stem, suffix in names:
                position = wanted.pop(stem, None)
                if position is None:
                    continue
                self._position[position] = chunk_id
                suffix_seen = suffix
                hits += 1
            remaining -= hits
            self._chunks.append(chunk)
            self._suffix_of_chunk.append(suffix_seen)
            if self._verbose and hits:
                print(
                    f"  {chunk.name}: {len(names)} файлов, из них нужных {hits}; "
                    f"осталось найти {remaining}",
                    flush=True,
                )

        if self._verbose:
            found = int((self._position >= 0).sum())
            print(
                f"индекс фотографий готов за {time.time() - started:.0f}s: "
                f"{found} из {len(wanted) + found} ключей найдено "
                f"в {len(self._chunks)} каталогах",
                flush=True,
            )
        return self

    def path_at(self, position: int) -> Path | None:
        if self._position is None or position >= len(self._position):
            return None
        chunk_id = int(self._position[position])
        if chunk_id < 0:
            return None
        return self._chunks[chunk_id] / (self._keys[position] + self._suffix_of_chunk[chunk_id])

    def get(self, key: str) -> Path | None:
        if not key or self._position is None:
            return None
        try:
            position = self._keys.index(key)
        except ValueError:
            return None
        return self.path_at(position)

    @property
    def found_count(self) -> int:
        return 0 if self._position is None else int((self._position >= 0).sum())

    def stats(self) -> dict[str, int]:
        return {"chunks": len(self._chunks), "found": self.found_count}

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.get(key) is not None

    def __len__(self) -> int:
        return self.found_count


class ImageIndex:
    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self.keys: list[str] = []

    def add(self, key: str) -> int:
        if not key:
            return -1
        idx = self._ids.get(key)
        if idx is None:
            idx = len(self.keys)
            self._ids[key] = idx
            self.keys.append(key)
        return idx

    def add_many(self, keys) -> np.ndarray:
        return np.fromiter((self.add(k) for k in keys), dtype=np.int64, count=len(keys))

    def __len__(self) -> int:
        return len(self.keys)


def dhash(image, size: int = DHASH_SIZE) -> np.uint64:
    from PIL import Image

    if size * size > 64:
        raise ValueError(f"хеш не влезает в 64 бита: size={size} даёт {size * size} бит")

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    packed = np.packbits(bits.flatten(), bitorder="big")
    return np.uint64(int.from_bytes(packed.tobytes().rjust(8, b"\x00"), "big"))


def dhash_many(
    keys: list[str],
    locator: ImageLocator,
    *,
    size: int = DHASH_SIZE,
    time_budget_s: float | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    import time

    hashes = np.zeros(len(keys), dtype=np.uint64)
    found = np.zeros(len(keys), dtype=bool)
    started = time.time()
    for i, key in enumerate(keys):
        if time_budget_s is not None and i % 1000 == 0 and time.time() - started > time_budget_s:
            if verbose:
                print(
                    f"бюджет исчерпан: хеши сняты для {i} из {len(keys)} фотографий",
                    flush=True,
                )
            break
        path = locator.path_at(i)
        if path is None:
            continue
        try:
            hashes[i] = dhash(path, size=size)
            found[i] = True
        except Exception:
            continue
        if verbose and i and i % 50_000 == 0:
            print(f"  хеши: {i}/{len(keys)} за {(time.time() - started) / 60:.0f} мин", flush=True)
    return hashes, found


def hamming(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(left.astype(np.uint64), right.astype(np.uint64))
    return np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1).astype(np.float32)


def _as_feature_tensor(raw):
    for attribute in ("image_embeds", "pooler_output"):
        value = getattr(raw, attribute, None)
        if value is not None:
            return value
    hidden = getattr(raw, "last_hidden_state", None)
    if hidden is not None:
        return hidden[:, 0]
    return raw


def encode_images(
    keys: list[str],
    locator: ImageLocator,
    *,
    model_name: str = DEFAULT_CLIP_MODEL,
    batch_size: int = 256,
    workers: int = 16,
    device: str | None = None,
    with_hashes: bool = True,
    time_budget_s: float | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import time

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    if device is None:
        device = pick_device(verbose=verbose)
    if verbose:
        print(f"кодирую {len(keys)} уникальных фотографий на {device}", flush=True)

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    dim = model.config.projection_dim

    out = np.zeros((len(keys), dim), dtype=np.float16)
    hashes = np.zeros(len(keys), dtype=np.uint64)
    found = np.zeros(len(keys), dtype=bool)
    started = time.time()
    stopped_at = None

    def load_one(position: int):
        path = locator.path_at(position)
        if path is None:
            return None
        try:
            image = Image.open(path)
            image.draft("RGB", (DRAFT_SIDE, DRAFT_SIDE))
            image.load()
        except Exception:
            return None
        digest = np.uint64(0)
        if with_hashes:
            try:
                digest = dhash(image)
            except Exception:
                digest = np.uint64(0)
        try:
            rgb = image.convert("RGB")
        except Exception:
            return None
        finally:
            image.close()
        return position, rgb, digest

    pool = ThreadPoolExecutor(max_workers=workers)

    with torch.inference_mode():
        for start in range(0, len(keys), batch_size):
            if time_budget_s is not None and time.time() - started > time_budget_s:
                stopped_at = start
                break
            chunk_positions = range(start, min(start + batch_size, len(keys)))
            images, positions = [], []
            for loaded in pool.map(load_one, chunk_positions):
                if loaded is None:
                    continue
                position, rgb, digest = loaded
                hashes[position] = digest
                images.append(rgb)
                positions.append(position)
            if not images:
                continue
            enc = processor(images=images, return_tensors="pt").to(device)
            features = _as_feature_tensor(model.get_image_features(**enc))
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            out[positions] = features.float().cpu().numpy().astype(np.float16)
            found[positions] = True
            for image in images:
                image.close()
            if verbose and start and start % (batch_size * 100) == 0:
                elapsed = time.time() - started
                print(f"  {start}/{len(keys)} за {elapsed / 60:.0f} мин", flush=True)

    pool.shutdown(wait=False)

    if verbose:
        elapsed = time.time() - started
        if stopped_at is not None:
            share = stopped_at / max(len(keys), 1)
            print(
                f"бюджет {time_budget_s / 60:.0f} мин исчерпан: закодировано "
                f"{stopped_at} из {len(keys)} ({share:.0%}), остальные пары "
                f"пойдут без визуальных признаков",
                flush=True,
            )
        else:
            print(
                f"закодировано {int(found.sum())} из {len(keys)} за {elapsed / 60:.0f} мин",
                flush=True,
            )

    return out, hashes, found


def image_pair_features(
    base_idx: np.ndarray,
    cand_idx: np.ndarray,
    *,
    hashes: np.ndarray | None = None,
    hash_found: np.ndarray | None = None,
    embeddings: np.ndarray | None = None,
    embedding_found: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    n = len(base_idx)
    out: dict[str, np.ndarray] = {}
    both_indexed = (base_idx >= 0) & (cand_idx >= 0)

    if hashes is not None:
        found = hash_found if hash_found is not None else np.ones(len(hashes), dtype=bool)
        pair_ok = both_indexed & found[base_idx] & found[cand_idx]
        distance = np.zeros(n, dtype=np.float32)
        if pair_ok.any():
            distance[pair_ok] = hamming(hashes[base_idx[pair_ok]], hashes[cand_idx[pair_ok]])
        bits = float(np.dtype(hashes.dtype).itemsize * 8)
        out["image_dhash_distance"] = np.where(pair_ok, distance / bits, 0.0).astype(np.float32)
        out["image_dhash_equal"] = (pair_ok & (distance == 0)).astype(np.float32)
        out["image_hash_pair_known"] = pair_ok.astype(np.float32)

    if embeddings is not None:
        found = (
            embedding_found
            if embedding_found is not None
            else np.ones(len(embeddings), dtype=bool)
        )
        pair_ok = both_indexed & found[base_idx] & found[cand_idx]
        cosine = np.zeros(n, dtype=np.float32)
        if pair_ok.any():
            left = embeddings[base_idx[pair_ok]].astype(np.float32)
            right = embeddings[cand_idx[pair_ok]].astype(np.float32)
            cosine[pair_ok] = np.clip((left * right).sum(axis=1), -1.0, 1.0)
        out["image_cosine"] = np.where(pair_ok, cosine, 0.0).astype(np.float32)
        out["image_embedding_pair_known"] = pair_ok.astype(np.float32)

    if not out:
        return {}

    out["image_missing"] = (~both_indexed).astype(np.float32)
    return out
