# Поиск дублей объявлений — Avito ML Cup 2025

Решение задачи [«Поиск дублей»](https://ods.ai/competitions/avitotechmlchallenge2025_2),
апрель–май 2025: для пары объявлений предсказать, является ли кандидат дублём базы.
Пара описывается 58 признаками сходства, поверх них учится LightGBM. Обучение — 1 879 555
пар, дублей 5.6 %.

| | PR-AUC | ROC-AUC | MAP |
|---|---|---|---|
| Baseline | 0.152 | 0.757 | — |
| Решение — LightGBM, OOF на 5 фолдах | **0.496** | 0.906 | 0.941 |

## Решение

| Модуль | Что делает |
|---|---|
| [`features.py`](src/features.py) | 58 парных признаков: Жаккар и containment заголовков и описаний, числа, `json_params`, цена, гео, категории |
| [`text.py`](src/text.py) | свёртка азбук для сравнения строк, починка гомоглифов перед трансформером |
| [`embeddings.py`](src/embeddings.py) | `rubert-tiny2` по 2.66 млн уникальных текстов, пары адресуют таблицу индексами |
| [`models.py`](src/models.py) | двухбашенная контрастная модель — обучаемая проекция над замороженным энкодером |
| [`images.py`](src/images.py) | индекс фотографий, dhash на перепост, CLIP на пересъёмку того же товара |
| [`validation.py`](src/validation.py) | GroupKFold по `group_id` |

## Вклад признаков

Абляция на одном фолде:

| Убрано | Потеря PR-AUC |
|---|---|
| текстовое сходство | 0.066 |
| цена, фото, гео | 0.023 |
| скор башен | 0.021 |
| сравнение фотографий | 0.015 |
| эмбеддинги | 0.005 |
| `json_params` | 0.004 |

## Что не сработало

| Эксперимент | PR-AUC | Почему хуже |
|---|---|---|
| [Кросс-энкодер](src/cross_encoder.py) — обе стороны пары через `[SEP]` | 0.300 | видит только текст, а кандидатов внутри кластера различают цена, категория и гео; смесь по рангам тоже не помогла |
| [Триплеты](src/triplets.py) — margin loss с трудными негативами | 0.459 | из 105 тыс. негативов 30 136 из той же базы, 74 903 из той же группы `param2`, случайных нет — башня свелась к ранжированию внутри кластера |

Для сравнения: решение даёт 0.496. Полные выводы прогонов —
[`reports/results.json`](reports/results.json) и
[`reports/results_triplet.json`](reports/results_triplet.json).

## Запуск

Ноутбук [`notebooks/avito-duplicates.ipynb`](notebooks/avito-duplicates.ipynb) запускается на
Kaggle с GPU: подключить [датасет соревнования](https://www.kaggle.com/datasets/chuvirla/avito-ml-cup-default-dataset)
и два набора с фотографиями ([train](https://www.kaggle.com/datasets/uhtiblya/train-images),
[test](https://www.kaggle.com/datasets/uhtiblya/test-images)), Save & Run All. На выходе —
`submission.csv` и `results.json`.

```
src/         модули решения
notebooks/   основной ноутбук и эксперимент с триплетами
reports/     полные выводы прогонов
```

## Лицензия

MIT — [LICENSE.txt](LICENSE.txt).
