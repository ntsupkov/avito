from __future__ import annotations

__all__ = ["pick_device", "reset_device_cache"]

_CACHED: str | None = None


def pick_device(verbose: bool = True) -> str:
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    import torch

    if not torch.cuda.is_available():
        _CACHED = "cpu"
        return _CACHED

    try:
        probe = torch.ones(8, 8, device="cuda")
        (probe @ probe).sum().item()
        _CACHED = "cuda"
        if verbose:
            print(f"устройство: cuda ({torch.cuda.get_device_name(0)})", flush=True)
    except Exception as exc:  # noqa: BLE001
        name = "неизвестна"
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            pass
        if verbose:
            print(f"видеокарта {name} непригодна ({exc}); считаю на CPU", flush=True)
        _CACHED = "cpu"

    return _CACHED


def reset_device_cache() -> None:
    global _CACHED
    _CACHED = None
