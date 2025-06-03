"""тексты обфусцированы: часть кириллицы подменена латинскими двойниками.

fold_confusables сводит обе азбуки к одной — когда сравниваю строки между собой.
restore_homoglyphs чинит слово в сторону его алфавита — перед подачей в модель.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

__all__ = [
    "fold_confusables",
    "restore_homoglyphs",
    "normalize",
    "tokenize",
    "jaccard",
    "char_ngrams",
    "normalize_vocabulary",
    "CONFUSABLES",
    "FOLD_PAIRS",
]

# пары «латинская — кириллическая», неразличимые в типичном шрифте. заглавные и строчные
# отдельно: К и K неразличимы, а к и k — вполне
CONFUSABLES: tuple[tuple[str, str], ...] = (
    ("a", "а"), ("c", "с"), ("e", "е"), ("o", "о"), ("p", "р"), ("x", "х"), ("y", "у"),
    ("A", "А"), ("B", "В"), ("C", "С"), ("E", "Е"), ("H", "Н"), ("K", "К"), ("M", "М"),
    ("O", "О"), ("P", "Р"), ("T", "Т"), ("X", "Х"), ("Y", "У"),
)

FOLD_PAIRS: tuple[tuple[str, str], ...] = (
    ("а", "a"), ("с", "c"), ("е", "e"), ("о", "o"), ("р", "p"), ("х", "x"), ("у", "y"),
    ("к", "k"), ("м", "m"), ("н", "h"), ("в", "b"), ("т", "t"),
)

_LATIN_AMBIGUOUS = {lat for lat, _ in CONFUSABLES}
_CYRILLIC_AMBIGUOUS = {cyr for _, cyr in CONFUSABLES}

_TO_CYRILLIC = str.maketrans({lat: cyr for lat, cyr in CONFUSABLES})
_TO_LATIN = str.maketrans({cyr: lat for lat, cyr in CONFUSABLES})
_FOLD = str.maketrans({cyr: lat for cyr, lat in FOLD_PAIRS})

_LATIN_RE = re.compile(r"[A-Za-z]")
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_PUNCT_RE = re.compile(r"[^\w\s]|_", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def fold_confusables(text: str) -> str:
    # ждёт нижний регистр. свёртка разрушающая, но починенных совпадений на порядки больше
    if not text:
        return ""
    return text.translate(_FOLD)


def _restore_token(token: str) -> str:
    # голосуют только однозначные буквы, у которых двойника нет
    has_real_cyrillic = False
    has_real_latin = False
    for ch in token:
        if ch not in _CYRILLIC_AMBIGUOUS and _CYRILLIC_RE.match(ch):
            has_real_cyrillic = True
        elif ch not in _LATIN_AMBIGUOUS and _LATIN_RE.match(ch):
            has_real_latin = True
    if has_real_cyrillic and not has_real_latin:
        return token.translate(_TO_CYRILLIC)
    if has_real_latin and not has_real_cyrillic:
        return token.translate(_TO_LATIN)
    # либо осмысленная смесь (iPhone), либо сплошные двойники (ecco, сор) — угадывать нечего
    return token


def restore_homoglyphs(text: str) -> str:
    # куpткa зимняя ecco -> куртка зимняя ecco, бренд остаётся латинским
    if not text:
        return ""
    return _WORD_RE.sub(lambda m: _restore_token(m.group(0)), text)


def normalize(text: str, *, fold: bool = True) -> str:
    # nfkc, нижний регистр, пунктуация в пробелы, пробелы схлопнуты; свёртка последней
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    if fold:
        text = fold_confusables(text)
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def tokenize(text: str, *, fold: bool = True) -> list[str]:
    # числа оставляю намеренно: в объявлениях они несут модель, размер и объём памяти —
    # самое различающее в парах вроде iphone 11 256 гб против iphone 11 128 гб
    return normalize(text, fold=fold).split()


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def char_ngrams(text: str, n: int = 3) -> set[str]:
    # устойчивы к опечаткам и другому порядку слов, значит ловят переписанные описания
    # там, где пословный жаккар уже проваливается
    s = normalize(text)
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def normalize_vocabulary(texts: Sequence[str], func) -> dict[str, str]:
    # уникальных токенов в корпусе на два порядка меньше, чем вхождений. считать по
    # словарю, а не по каждому вхождению — это секунды против часов на миллионах строк
    vocab: dict[str, str] = {}
    for text in texts:
        if not text:
            continue
        for token in _WORD_RE.findall(text):
            if token not in vocab:
                vocab[token] = func(token)
    return vocab
