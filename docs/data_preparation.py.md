# `data_preparation.py`

## Назначение

Модуль превращает Excel/CSV-выгрузку иерархической витрины в нормализованную
историю и полную панель `segment_id × cal_date`. Побочных эффектов записи нет.

## Публичные функции

| Функция | Роль |
|---|---|
| `normalize_dim_value` | `None`, `NaN`, пустая строка → `None`; остальное → trimmed string |
| `segment_id_from_row` | Технический ID из всех dimensions, пропуск = `∅` |
| `build_segment_key_and_level` | Читаемый ключ, уровень и вычисленная глубина |
| `candidate_covers_atomic` | Проверка, совместим ли родитель со значениями атома |
| `infer_anomaly_dimension_columns` | Валидирует и возвращает явно заданные dimensions |
| `load_history_table` | Загрузка, типизация, валидация total-слоя |
| `build_full_week_grid` | Полная сетка сегментов и недель |

## `load_history_table`

Поддерживает `.xlsx`, `.xls`, `.csv`.

Обязательные колонки: `cal_date`, `slice_depth`, `gmv`.

Последовательность:

1. Читает файл.
2. При наличии `period` удаляет служебную строку с `period == "string"` и
   применяет фильтр периода.
3. Приводит три обязательные колонки к числам; невалидные строки удаляет.
4. Приводит доступные `tx/au/am/aov/tpm/freq` к числам.
5. Определяет dimensions и нормализует их значения.
6. Добавляет `segment_id`, `segment_key`, `segment_level`.
7. Сверяет вычисленную глубину сегмента с входным `slice_depth`, валидирует
   `segment_key` и согласованность metadata сегмента между неделями.
8. Проверяет уникальность `segment_id × cal_date`.
9. По `slice_depth=0` строит список недель и проверяет календарь.

Dimensions зафиксированы в `config.DIM_COLUMNS`. Любая другая колонка входного Excel
считается технической, входит в `ANOMALY_TECH_COLUMNS` в рамках конкретной
выгрузки и не участвует в `segment_id`, `segment_key` или coverage.

Жёсткие проверки:

- total-слой существует;
- ключ `segment_id × cal_date` уникален;
- входной `slice_depth` равен числу заполненных dimensions; `segment_key` и
  `segment_level` согласованы с ними;
- metadata (`slice_depth`, ключ, уровень и dimensions) одинакова во всех
  неделях одного `segment_id`;
- есть минимум четыре недели;
- шаг между `cal_date` равен 7;
- total GMV строго положителен.

`dates` строится по агрегированному total-слою, поэтому именно total определяет
временную ось всего дальнейшего анализа.

## Идентификаторы сегмента

Для dimensions `["geo", "products"]`:

| Значения | `segment_id` | `segment_key` | `segment_level` |
|---|---|---|---|
| `None`, `None` | `∅|∅` | `ИТОГО` | `ИТОГО` |
| `РФ`, `None` | `РФ|∅` | `geo=РФ` | `geo` |
| `РФ`, `QR` | `РФ|QR` | `geo=РФ × products=QR` | `geo × products` |

Вычисленная глубина ключа строго сверяется с входным `slice_depth` на этапе
`load_history_table`. Вход отклоняется до расчёта аномалий, если, например,
`geo=РФ × products=QR` имеет `slice_depth = 1`, либо metadata одного
`segment_id` различается между неделями.

## `build_full_week_grid`

1. Берёт одну metadata-строку на `segment_id`.
2. Переносит предагрегированные YQL значения `gmv`, `tx`, `au`, `am`, `aov`, `tpm`, `freq` без повторной агрегации.
3. Строит декартову сетку всех сегментов и всех total-недель.
4. Отсутствующие GMV и метрики заменяет нулём.
5. Добавляет `row_missing_in_source` по результату соединения с исходными строками.

## Критичные семантические решения

- Отсутствующая строка источника становится фактическим нулём для статистики.
  Отличить её можно только по `row_missing_in_source`.
- `pred_insight.yql` формирует одну строку на `period × cal_date × segment_id`.
  После фильтрации выбранного `period` ключ `segment_id × cal_date` обязан быть
  уникальным. Дубликаты считаются нарушением входного контракта и отклоняются
  в `load_history_table`; Python не исправляет их повторной агрегацией.
- Новая колонка Excel не меняет состав dimensions: она считается технической и
  не становится частью `segment_id`. Для изменения сегментации нужно явно
  изменить `DIM_COLUMNS` и синхронно обновить входной контракт и тесты coverage.
- Перед `drop_duplicates` проверяется согласованность metadata одного
  `segment_id`, поэтому `build_full_week_grid` не может молча выбрать первую
  из противоречивых строк.

## Связи

- Входной контракт задаёт [`pred_insight.yql.md`](pred_insight.yql.md).
- Результат `load_history_table` передаётся в `build_full_week_grid`.
- Панель потребляет `anomaly_scoring.build_anomaly_candidates`.
- Известные риски №1 и №2 описаны в [`refactoring-findings.md`](refactoring-findings.md).
